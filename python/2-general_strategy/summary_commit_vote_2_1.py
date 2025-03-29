import os
import json
import sys
import time
import subprocess
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI
# 对每个commit生成的总结次数
NUM_GENERATIONS = 5
# LLM 的最大重试次数
MAX_LLM_RETRIES = 3
# 温度范围设置
TEMPERATURE_MIN = 0.1  # 温度下限
TEMPERATURE_MAX = 0.3  # 温度上限
# 如果commit已包含要生成的字段，是否重新生成
# False：使用已有结果并按需补足到 NUM_GENERATIONS 个
# True：强制重新生成所有结果
REGENERATE_EXISTING = True
# 指定要处理的JSON文件名
INPUT_FILE_PATH = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_test_new/partial_commit_test_dsv3_2.json")
LLM_MODEL = config.xmcp_deepseek_model
# 最大并行线程数
MAX_WORKERS = 32
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
    使用随机生成的temperature值
    """
    # 生成随机温度值
    temperature = random.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX)
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistent."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
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
    处理单个commit：基于Self-consistency voting生成多个优化策略总结，然后进行投票
    """
    # 检查commit是否已经有部分结果
    existing_summaries = commit.get("optimization_summary", [])
    existing_generic = commit.get("is_generic_optimization", [])
    
    # 如果不重新生成且已有最终结果，则直接跳过
    if not REGENERATE_EXISTING and "optimization_summary_final" in commit and "is_generic_optimization_final" in commit:
        return {
            "optimization_summary": existing_summaries,
            "is_generic_optimization": existing_generic,
            "optimization_summary_final": commit.get("optimization_summary_final", ""),
            "is_generic_optimization_final": commit.get("is_generic_optimization_final", False),
            "skipped": True
        }
    
    repo_name = commit.get('repository_name', '')
    commit_hash = commit.get('hash', '')
    commit_message = commit.get('message', '')
    modified_files = ", ".join(commit.get('modified_files', []))
    modified_func = ", ".join(commit.get('modified_func', []))
    
    # 获取代码差异
    diff_content = get_diff_by_commit(repo_name, commit_hash)
    
    # 构建基础提示
    base_prompt = (
        "Please analyze the following commit that implements performance optimization. "
        "Your task is to:\n\n"
        "1. Provide a ONE-SENTENCE summary in English about what performance optimization strategy was used. "
        "Focus on summarizing the strategy at a higher conceptual level instead of code-specific details. "
        "Describe how the optimization generally improves performance rather than explaining specific code changes.\n"
        "2. Determine if this is a GENERIC optimization strategy (answer only with 'true' or 'false').\n\n"
        
        "A generic optimization strategy has the following characteristics:\n"
        "- It is a commonly used performance optimization strategy that programmers can easily understand why it improves performance\n"
        "- In most cases, applying this strategy should lead to performance improvements or at least not decrease performance\n"
        "- Programmers don't need comprehensive knowledge of the entire codebase to understand why this strategy works\n"
        "- The strategy doesn't require satisfying many code-specific preconditions to be applicable\n" 
        "- Programmers can determine if the strategy is applicable using relatively simple criteria, without needing to know many specific details about the code\n\n"
        
        "Examples of generic optimization strategies:\n"
        "- Changing value-based loop iteration to reference-based iteration to reduce copy overhead\n"
        "- Reordering conditions in if-statements with multiple conditions connected by AND operators to check simple conditions first\n"
        "- Using more efficient data structures or algorithms for specific operations\n\n"
        
        "Examples of NON-generic optimization strategies (these are context-specific):\n"
        "- Modifying random test sections to reduce the number of random iterations (e.g., changing loop count from 10 to 30)\n"
        "- Adding a conditional check before calling a deletion operation to avoid unnecessary calls in a specific scenario\n"
        "- Moving time-consuming work outside of a mutex lock section for a specific synchronized block\n"
        "- Distinguishing between data blocks and non-data blocks to only attempt compression/decompression on the former\n\n"
        
        f"Repository: {repo_name}\n"
        f"Commit Hash: {commit_hash}\n"
        f"Commit Message: {commit_message}\n"
        f"Modified Files: {modified_files}\n"
        f"Modified Functions: {modified_func}\n\n"
        f"Code Changes:\n{diff_content}\n\n"
        
        "Please format your response exactly as follows:\n"
        "Summary: [your one-sentence summary of the optimization strategy at a higher conceptual level]\n"
        "Generic: [true or false]"
    )
    
    # 用于存储多次尝试的结果
    summaries = existing_summaries.copy() if not REGENERATE_EXISTING else []
    generic_results = existing_generic.copy() if not REGENERATE_EXISTING else []
    
    # 生成需要的次数，如果已有部分结果且不需要重新生成，则只生成剩余需要的次数
    remaining_generations = NUM_GENERATIONS - len(summaries) if not REGENERATE_EXISTING else NUM_GENERATIONS
    
    # 生成指定次数的结果
    for i in range(remaining_generations):
        # 在prompt开头添加一个id，避免缓存命中
        prompt = f"Query ID: {i+1} for commit {commit_hash}\n\n" + base_prompt
        
        retry_count = 0
        while retry_count < MAX_LLM_RETRIES:
            response = call_llm(prompt)  # 每次调用都会使用随机temperature
            if response is not None:
                summary, is_generic = parse_llm_response(response)
                if summary:  # 确保解析出了有效的总结
                    summaries.append(summary)
                    generic_results.append(is_generic)
                    break
            retry_count += 1
            time.sleep(1)  # 短暂等待后重试
    
    # 如果没有生成任何有效结果，返回None
    if not summaries:
        return None
    
    # 通过Self-consistency voting选择最终结果
    # 1. 对summary进行投票 - 选择最相似的群组
    similarity_matrix = np.zeros((len(summaries), len(summaries)))
    for i in range(len(summaries)):
        for j in range(len(summaries)):
            if i != j:
                similarity_matrix[i][j] = calculate_similarity(summaries[i], summaries[j])
    
    # 计算每个summary的平均相似度
    avg_similarities = np.mean(similarity_matrix, axis=1)
    best_summary_index = np.argmax(avg_similarities)
    final_summary = summaries[best_summary_index]
    
    # 2. 对generic结果进行多数投票
    generic_counter = Counter(generic_results)
    final_is_generic = generic_counter.most_common(1)[0][0]
    
    return {
        "optimization_summary": summaries,
        "is_generic_optimization": generic_results,
        "optimization_summary_final": final_summary,
        "is_generic_optimization_final": final_is_generic,
        "skipped": False
    }
def main():
    """
    主函数：处理指定的partial_commit文件
    """
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"错误：指定的文件 {INPUT_FILE_PATH} 不存在")
        return
    
    print(f"\n处理文件: {INPUT_FILE_PATH}")
    print(f"每个commit生成 {NUM_GENERATIONS} 个总结进行投票")
    print(f"使用随机temperature: {TEMPERATURE_MIN} - {TEMPERATURE_MAX}")
    
    # 读取commit数据
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        commits = json.load(f)
    
    print(f"文件中包含 {len(commits)} 个commits")
    
    # 检查有多少commit已经包含结果
    existing_results = sum(1 for commit in commits 
                           if "optimization_summary_final" in commit 
                           and "is_generic_optimization_final" in commit)
    
    if existing_results > 0:
        if REGENERATE_EXISTING:
            print(f"找到 {existing_results} 个已有结果的commit，将全部重新生成")
        else:
            print(f"找到 {existing_results} 个已有结果的commit，将使用现有结果（如果数量不足会补充生成）")
    
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
                    
                    # 删除不需要的旧格式字段
                    for field in ["all_summaries", "all_generic_results"]:
                        if field in commit:
                            del commit[field]
                    
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