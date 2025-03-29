import os
import json
import sys
import re
import difflib
import concurrent.futures
import datetime
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 全局变量
# LLM模型名称（用于定位优化结果目录）
MODEL_NAME = "qwen-max"
# API调用的模型名称
MODEL_API_NAME = config.xmcp_qwen_model if MODEL_NAME == "qwen-max" else config.xmcp_deepseek_model
# 温度设置
TEMPERATURE = 0.1
# 输入：策略适用性评估文件
INPUT_ASSESSMENT_FILE = os.path.join(config.root_path, "python/2-general_strategy/result/kmeans_40000_dsv3_true_14000/order_sum_11_100_10_dsv3_full_16_appone_qwenmax.json")
# 输出：优化结果评估文件
OUTPUT_ASSESSMENT_FILE = os.path.join(config.root_path, "python/2-general_strategy/result/kmeans_40000_dsv3_true_14000/commit_result/order_sum_11_100_10_dsv3_full_16_appone_qwenmax/result.json")
# 知识库路径
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")
# LLM优化结果路径
LLM_RESULT_PATH = os.path.join(config.root_path, "python/2-general_strategy/result/kmeans_40000_dsv3_true_14000/commit_result/order_sum_11_100_10_dsv3_full_16_appone_qwenmax/")
# 并行处理的最大线程数
MAX_WORKERS = 32
# 每个commit评估的优化尝试次数
OPTIMIZATION_ATTEMPTS = 3

def get_client():
    """初始化OpenAI客户端"""
    try:
        client = OpenAI(
            base_url=config.xmcp_base_url,
            api_key=config.xmcp_api_key,
        )
        return client
    except Exception as e:
        raise RuntimeError(f"无法初始化 OpenAI 客户端: {e}")

def get_file_content(file_path):
    """获取文件内容"""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def generate_diff(before_code, after_code):
    """生成两段代码之间的差异"""
    before_lines = before_code.splitlines()
    after_lines = after_code.splitlines()
    
    differ = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile='before',
        tofile='after',
        lineterm=''
    )
    
    return '\n'.join(differ)

def assess_optimization_similarity(client, repo_name, commit_hash, attempt_number):
    """评估指定尝试编号的LLM优化结果与实际commit差异的相似性"""
    result = {
        "repo_name": repo_name,
        "commit_hash": commit_hash,
        "model_name": MODEL_NAME,
        "attempt_number": attempt_number,
        "is_similar": False,
        "error": None
    }
    
    # 构建文件路径
    before_func_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash, "before_func.txt")
    after_func_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash, "after_func.txt")
    diff_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash, "diff.txt")
    
    # 直接使用带模型名的路径结构
    optimized_func_path = os.path.join(LLM_RESULT_PATH, repo_name, commit_hash, MODEL_NAME, f"{attempt_number}.txt")
    
    # 检查优化结果文件是否存在
    if not os.path.exists(optimized_func_path):
        result["error"] = f"Missing optimization result file: {optimized_func_path}"
        return result
    
    # 检查知识库文件是否存在
    if not all(os.path.exists(path) for path in [before_func_path, after_func_path, diff_path]):
        missing_files = []
        for path in [before_func_path, after_func_path, diff_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        result["error"] = f"Missing knowledge base files: {', '.join(missing_files)}"
        return result
    
    # 读取文件内容
    before_func = get_file_content(before_func_path)
    after_func = get_file_content(after_func_path)
    original_diff = get_file_content(diff_path)
    optimized_func = get_file_content(optimized_func_path)
    
    if not all([before_func, after_func, original_diff, optimized_func]):
        result["error"] = "Failed to read one or more files"
        return result
    
    # 生成LLM优化结果与原始代码的差异
    generated_diff = generate_diff(before_func, optimized_func)
    
    # 添加唯一标识符，仅使用尝试编号
    unique_id = f"[Attempt: {attempt_number}]"
    
    # 构建提示，让LLM比较两个差异
    prompt = f"""{unique_id}

Compare the following two code changes and determine if they are implementing similar optimizations:

## Original Commit Diff (real change made by developers):
```
{original_diff}
```

## Generated Optimization Diff (from LLM):
```
{generated_diff}
```

Focus on:
1. Are both changes implementing the same core optimization idea?
2. Do both changes modify similar parts of the code?
3. Do both changes achieve the same performance improvement goal?

Answer with SIMILAR if the LLM optimization captures the key aspects of the original commit's optimization intent, even if implementation details differ slightly.
Answer with DIFFERENT if the LLM optimization takes a fundamentally different approach or misses the core optimization goal.
Provide a brief explanation of your reasoning.
"""
    
    messages = [{"role": "user", "content": prompt}]
    
    # 调用LLM进行比较
    try:
        response = client.chat.completions.create(
            model=MODEL_API_NAME,
            messages=messages,
            temperature=TEMPERATURE
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # 判断是否相似
        is_similar = "SIMILAR" in response_text.upper()
        result["is_similar"] = is_similar
        result["llm_assessment"] = response_text
        
    except Exception as e:
        result["error"] = f"API error: {e}"
    
    return result

def process_commit_with_attempt(args):
    """处理单个commit的指定尝试编号的评估"""
    client, repo_name, commit_hash, attempt_number = args
    return assess_optimization_similarity(client, repo_name, commit_hash, attempt_number)

def main():
    # 加载策略适用性评估文件
    print(f"加载策略适用性评估文件: {INPUT_ASSESSMENT_FILE}")
    try:
        with open(INPUT_ASSESSMENT_FILE, 'r', encoding='utf-8') as f:
            assessment_data = json.load(f)
    except Exception as e:
        print(f"无法读取评估文件: {e}")
        sys.exit(1)
    
    # 收集需要评估的commits
    commits_to_process = []
    
    # 从新格式的JSON中提取commit信息
    for cluster_id, commits in assessment_data.get("cluster_assessments", {}).items():
        for commit in commits:
            commit_hash = commit.get("commit_hash")
            repo_name = commit.get("repository")
            is_applicable = commit.get("voting_result", False)
            
            # 只处理标记为适用的commit
            if is_applicable and commit_hash and repo_name:
                # 对每个commit评估所有尝试
                for attempt in range(1, OPTIMIZATION_ATTEMPTS + 1):
                    commits_to_process.append((repo_name, commit_hash, attempt))
    
    print(f"找到 {len(commits_to_process)} 个优化结果需要评估 (来自 {len(commits_to_process) // OPTIMIZATION_ATTEMPTS} 个commit)")
    
    # 初始化OpenAI客户端
    client = get_client()
    
    # 准备并行处理参数
    args_list = [(client, repo_name, commit_hash, attempt) for repo_name, commit_hash, attempt in commits_to_process]
    
    # 并行处理所有评估任务
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(
            executor.map(process_commit_with_attempt, args_list), 
            total=len(args_list), 
            desc="评估优化相似性"
        ):
            results.append(result)
    
    # 按commit分组结果
    commit_results = {}
    for result in results:
        key = (result["repo_name"], result["commit_hash"])
        if key not in commit_results:
            commit_results[key] = []
        commit_results[key].append(result)
    
    # 找出每个commit中最好的尝试（如果有相似的优先选相似的，否则选没有错误的）
    best_attempts = []
    for commit_group in commit_results.values():
        # 首先尝试找到相似的结果
        similar_attempts = [r for r in commit_group if r["is_similar"] and not r["error"]]
        if similar_attempts:
            best_attempts.append(similar_attempts[0])  # 选择第一个相似的结果
        else:
            # 如果没有相似的，找一个没有错误的结果
            valid_attempts = [r for r in commit_group if not r["error"]]
            if valid_attempts:
                best_attempts.append(valid_attempts[0])  # 选择第一个有效的结果
            else:
                # 如果全都有错误，选择第一个
                best_attempts.append(commit_group[0])
    
    # 汇总结果并保存
    output_data = {
        "metadata": {
            "model": MODEL_NAME,
            "total_assessments": len(results),
            "total_commits": len(commit_results),
            "similar_count": sum(1 for r in results if r["is_similar"]),
            "different_count": sum(1 for r in results if not r["is_similar"] and not r["error"]),
            "error_count": sum(1 for r in results if r["error"]),
            "generated_at": datetime.datetime.now().isoformat()
        },
        "assessments": results,  # 所有评估结果
        "best_attempts": best_attempts  # 每个commit的最佳尝试
    }
    
    # 计算成功率
    total_valid = output_data["metadata"]["similar_count"] + output_data["metadata"]["different_count"]
    if total_valid > 0:
        output_data["metadata"]["similarity_rate"] = output_data["metadata"]["similar_count"] / total_valid
    else:
        output_data["metadata"]["similarity_rate"] = 0
    
    # 计算每个commit的最佳尝试中有多少是相似的
    best_similar_count = sum(1 for r in best_attempts if r["is_similar"])
    total_valid_best = sum(1 for r in best_attempts if not r["error"])
    output_data["metadata"]["best_attempts_similar_count"] = best_similar_count
    if total_valid_best > 0:
        output_data["metadata"]["best_attempts_similarity_rate"] = best_similar_count / total_valid_best
    else:
        output_data["metadata"]["best_attempts_similarity_rate"] = 0
    
    # 保存结果
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(OUTPUT_ASSESSMENT_FILE), exist_ok=True)
        
        with open(OUTPUT_ASSESSMENT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存到 {OUTPUT_ASSESSMENT_FILE}")
        print(f"总评估结果数: {output_data['metadata']['total_assessments']}, 总commit数: {output_data['metadata']['total_commits']}")
        print(f"相似结果数: {output_data['metadata']['similar_count']}, " 
              f"不同结果数: {output_data['metadata']['different_count']}, "
              f"错误结果数: {output_data['metadata']['error_count']}")
        if total_valid > 0:
            print(f"所有结果相似率: {output_data['metadata']['similarity_rate']:.2%}")
        if total_valid_best > 0:
            print(f"每个commit最佳尝试相似率: {output_data['metadata']['best_attempts_similarity_rate']:.2%}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == "__main__":
    main()
