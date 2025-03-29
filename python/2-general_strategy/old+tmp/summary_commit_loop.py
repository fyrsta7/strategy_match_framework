import os
import json
import sys
import time
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 最大并行线程数
MAX_WORKERS = 32

# 如果commit已包含要生成的字段，是否重新生成
# False：跳过已有结果的commit
# True：强制重新生成所有结果
REGENERATE_EXISTING = True

# 指定要处理的JSON文件名
INPUT_FILE_PATH = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_70330.json")

# LLM重试相关配置
MAX_LLM_RETRIES = 5  # 最大重试次数
SIMILARITY_THRESHOLD = 0.8  # 相似度阈值

# LLM温度
TEMPERATUE = 0.2

# 初始化句子转换模型用于计算相似度
model_path = os.path.join(config.root_path, "models/all-MiniLM-L6-v2")
sentence_model = SentenceTransformer(model_path)

# 初始化OpenAI客户端
client = OpenAI(
    base_url=config.xmcp_base_url,
    api_key=config.xmcp_api_key_unlimit,
)

def call_llm(prompt):
    """
    通用的 llm 调用函数，返回回复内容
    """
    try:
        response = client.chat.completions.create(
            model=config.xmcp_deepseek_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistent."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.2
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"调用LLM出错: {e}")
        time.sleep(3)  # 发生错误时等待更长时间
        return None

def get_diff_by_commit(repo_name, commit_hash):
    """
    获取指定commit的代码差异
    """
    try:
        repo_path = os.path.join(config.root_path, 'benchmark', repo_name)
        
        # 检查仓库目录是否存在
        if not os.path.exists(repo_path):
            return f"Repository directory not found: {repo_path}"
        
        # 执行git show命令获取diff
        cmd = f"cd {repo_path} && git show {commit_hash} -p"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            return f"Error fetching diff: {result.stderr}"
        
        return result.stdout
    except Exception as e:
        return f"Error getting diff: {str(e)}"

def calculate_similarity(text1, text2):
    """
    计算两段文本的相似度
    """
    # 使用sentence-transformers编码文本
    embedding1 = sentence_model.encode([text1])[0]
    embedding2 = sentence_model.encode([text2])[0]
    
    # 计算余弦相似度
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

def parse_llm_response(response):
    """
    解析LLM的回复，提取出summary和generic字段
    """
    summary = ""
    is_generic = False
    
    for line in response.split('\n'):
        line = line.strip()
        if line.lower().startswith("summary:"):
            summary = line[len("summary:"):].strip()
        elif line.lower().startswith("generic:"):
            generic_value = line[len("generic:"):].strip().lower()
            is_generic = generic_value == "true"
    
    return summary, is_generic

def process_commit(commit, input_file_name):
    """
    处理单个commit：生成优化策略总结和判断是否为通用优化
    """
    # 检查commit是否已经包含所需字段
    if not REGENERATE_EXISTING and "optimization_summary_final" in commit and "is_generic_optimization_final" in commit:
        return {
            "optimization_summary": commit.get("optimization_summary", []),
            "is_generic_optimization": commit.get("is_generic_optimization", []),
            "optimization_summary_final": commit.get("optimization_summary_final", []),
            "is_generic_optimization_final": commit.get("is_generic_optimization_final", []),
            "skipped": True
        }
    
    repo_name = commit.get('repository_name', '')
    commit_hash = commit.get('hash', '')
    commit_message = commit.get('message', '')
    modified_files = ", ".join(commit.get('modified_files', []))
    modified_func = ", ".join(commit.get('modified_func', []))
    
    # 获取代码差异
    diff_content = get_diff_by_commit(repo_name, commit_hash)
    
    # 构建提示
    prompt = (
        "Please analyze the following commit that implements performance optimization. "
        "Your task is to:\n\n"
        "1. Provide a ONE-SENTENCE summary in English about what optimization strategy was used.\n"
        "2. Determine if this is a GENERIC optimization strategy (answer only with 'true' or 'false').\n\n"
        
        "A generic optimization strategy has the following characteristics:\n"
        "- Low context requirements: Can be applied without detailed knowledge of the entire codebase's functionality or specific performance bottlenecks\n"
        "- Local reasoning: Can be assessed by examining only the function or file being optimized, without needing broader context\n"
        "- Slightly above compiler optimizations: Could theoretically be implemented by compilers but typically isn't\n"
        "- Generally preserves correctness: Provides reasonable guarantees that the optimized code maintains the same behavior\n"
        "- Broadly applicable: Can be applied across different codebases in similar situations\n\n"
        
        "Examples of generic optimization strategies:\n"
        "- Changing value-based loop iteration to reference-based iteration to reduce copy overhead\n"
        "- Reordering conditions in if-statements with multiple conditions connected by AND operators\n\n"
        
        "Examples of NON-generic optimization strategies (these are context-specific):\n"
        "- Modifying random test sections to reduce the number of random iterations (e.g., changing loop count from 10 to 30)\n"
        "- Adding a conditional check before calling a deletion operation to avoid unnecessary calls\n"
        "- Moving time-consuming work outside of a mutex lock section\n"
        "- Distinguishing between data blocks and non-data blocks to only attempt compression/decompression on the former\n\n"
        
        f"Repository: {repo_name}\n"
        f"Commit Hash: {commit_hash}\n"
        f"Commit Message: {commit_message}\n"
        f"Modified Files: {modified_files}\n"
        f"Modified Functions: {modified_func}\n\n"
        f"Code Changes:\n{diff_content}\n\n"
        
        "Please format your response exactly as follows:\n"
        "Summary: [your one-sentence summary of the optimization strategy]\n"
        "Generic: [true or false]"
    )
    
    # 用于存储多次尝试的结果
    summaries = []
    generic_results = []
    
    # 至少尝试两次，为了计算初始相似度
    for i in range(2):
        response = call_llm(prompt)
        if response is None:
            continue
            
        summary, is_generic = parse_llm_response(response)
        summaries.append(summary)
        generic_results.append(is_generic)
    
    # 计算前两次结果的相似度，如果不够高就继续尝试
    retry_count = 2
    while retry_count < MAX_LLM_RETRIES:
        # 检查是否已找到相似度足够高的结果
        found_reliable_result = False
        
        # 计算目前所有结果之间的相似度
        for i in range(len(summaries)):
            for j in range(i+1, len(summaries)):
                if calculate_similarity(summaries[i], summaries[j]) >= SIMILARITY_THRESHOLD:
                    found_reliable_result = True
                    break
            if found_reliable_result:
                break
                
        # 如果找到了相似度足够高的结果，就停止尝试
        if found_reliable_result:
            break
            
        # 继续调用LLM
        response = call_llm(prompt)
        if response is None:
            retry_count += 1
            continue
            
        summary, is_generic = parse_llm_response(response)
        summaries.append(summary)
        generic_results.append(is_generic)
        retry_count += 1
    
    # 选择最可靠的结果
    if not summaries:
        return None
        
    # 为每个summary计算与其他summaries的平均相似度
    avg_similarities = []
    for i in range(len(summaries)):
        similarities = []
        for j in range(len(summaries)):
            if i != j:
                similarities.append(calculate_similarity(summaries[i], summaries[j]))
        avg_similarities.append(sum(similarities) / len(similarities) if similarities else 0)
    
    # 选择平均相似度最高的summary作为final结果
    best_summary_index = avg_similarities.index(max(avg_similarities))
    final_summary = summaries[best_summary_index]
    
    # 对于generic结果，采用多数投票
    generic_votes = sum(1 for g in generic_results if g)
    final_is_generic = generic_votes > len(generic_results) / 2
    
    return {
        "optimization_summary": summaries,
        "is_generic_optimization": generic_results,
        "optimization_summary_final": [final_summary],
        "is_generic_optimization_final": [final_is_generic],
        "skipped": False
    }

def main():
    """
    主函数：处理指定的partial_commit文件
    """
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"错误：指定的文件 {INPUT_FILE_PATH} 不存在")
    
    print(f"\n处理文件: {INPUT_FILE_PATH}")
    
    # 读取commit数据
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        commits = json.load(f)
    
    print(f"文件中包含 {len(commits)} 个commits")
    
    # 检查有多少commit已经包含结果
    existing_results = sum(1 for commit in commits if "optimization_summary_final" in commit and "is_generic_optimization_final" in commit)
    if existing_results > 0:
        if REGENERATE_EXISTING:
            print(f"找到 {existing_results} 个已有结果的commit，将重新生成")
        else:
            print(f"找到 {existing_results} 个已有结果的commit，将跳过")
    
    # 统计变量
    stats = {"total": len(commits), "processed": 0, "skipped": 0, "failed": 0}
    
    # 并行处理所有commits
    updated_commits = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 创建任务映射
        future_to_commit = {}
        for commit in commits:
            future = executor.submit(process_commit, commit, INPUT_FILE_PATH)
            future_to_commit[future] = commit
        
        # 处理结果
        for future in tqdm(as_completed(future_to_commit), total=len(future_to_commit), desc="处理commits"):
            commit = future_to_commit[future]
            try:
                result = future.result()
                if result:
                    # 更新commit数据
                    commit["optimization_summary"] = result["optimization_summary"]
                    commit["is_generic_optimization"] = result["is_generic_optimization"]
                    commit["optimization_summary_final"] = result["optimization_summary_final"]
                    commit["is_generic_optimization_final"] = result["is_generic_optimization_final"]
                    
                    # 删除不需要的字段
                    if "all_summaries" in commit:
                        del commit["all_summaries"]
                    if "all_generic_results" in commit:
                        del commit["all_generic_results"]
                    
                    if result.get("skipped", False):
                        stats["skipped"] += 1
                    else:
                        stats["processed"] += 1
                else:
                    print(f"处理commit {commit.get('hash', '未知')} 失败")
                    stats["failed"] += 1
            except Exception as e:
                print(f"处理commit时发生异常: {e}")
                stats["failed"] += 1
            
            updated_commits.append(commit)
    
    # 保存更新后的数据
    with open(INPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_commits, f, ensure_ascii=False, indent=4)
    
    print(f"\n处理统计:")
    print(f"总计: {stats['total']} 个commit")
    print(f"成功处理: {stats['processed']} 个")
    print(f"跳过已有结果: {stats['skipped']} 个")
    print(f"处理失败: {stats['failed']} 个")
    print(f"成功更新文件: {INPUT_FILE_PATH}")

if __name__ == "__main__":
    main()