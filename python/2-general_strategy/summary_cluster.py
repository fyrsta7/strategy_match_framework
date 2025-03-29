import json
import os
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import re
from collections import Counter
import numpy as np
import sys
from sentence_transformers import SentenceTransformer, util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 模型配置
LLM_MODEL_NAME = "dsv3"
LLM_MODEL = config.xmcp_deepseek_model if LLM_MODEL_NAME == "dsv3" else config.xmcp_qwen_model
TEMPERATURE = 0.1

# 控制使用哪些信息
USE_DIFF_INFO = True  # 是否使用 diff 信息
USE_COMMIT_MESSAGE = True  # 是否使用 commit message
# 使用哪个版本的应用条件 prompt (True=新版本，False=旧版本)
USE_PROMPT = False
USE_DIFF_INFO_STR = "1" if USE_DIFF_INFO else "0"
USE_COMMIT_MESSAGE_STR = "1" if USE_COMMIT_MESSAGE else "0"
USE_PROMPT_STR = "1" if USE_PROMPT else "0"

# 每个聚类最多使用的commit数量
MAX_COMMITS_PER_CLUSTER = 10
# diff信息的最大长度
MAX_DIFF_LENGTH = 2000
# 是否跳过需要截断的commit (True=跳过截断的commit使用后面的，False=使用截断的)
SKIP_TRUNCATED_DIFF = True
# 使用 self-consistency voting 的重复次数
CONSISTENCY_REPEATS = 5
# 只处理 commit 数量大于等于此阈值的聚类
threshold = 11

# Sentence Transformer 模型路径
SENTENCE_MODEL_PATH = os.path.join(config.root_path, "models/all-MiniLM-L6-v2")
DIR_NAME = "kmeans_40000_dsv3_true_14000/"
FILE_NAME = "order"
INPUT_FILE = config.root_path + "python/2-general_strategy/result/" + f"{DIR_NAME}/{FILE_NAME}.json"
OUTPUT_FILE = config.root_path + "python/2-general_strategy/result/" + f"{DIR_NAME}/{FILE_NAME}_sum_{threshold}_{USE_DIFF_INFO_STR}{USE_COMMIT_MESSAGE_STR}{USE_PROMPT_STR}_{MAX_COMMITS_PER_CLUSTER}_{LLM_MODEL_NAME}.json"

# 并行处理的最大线程数
MAX_WORKERS = 16

# 初始化OpenAI客户端
client = OpenAI(
    base_url=config.xmcp_base_url,
    api_key=config.xmcp_api_key_unlimit,
)

# 初始化 sentence transformer 模型 (用于计算相似度)
sentence_model = SentenceTransformer(SENTENCE_MODEL_PATH)

# 自定义JSON编码器，处理numpy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def call_llm(messages, temperature=TEMPERATURE, call_id=None):
    """
    通用的 llm 对话函数，接收消息列表，返回回复内容
    添加call_id参数来标识每次调用
    """
    try:
        # 如果提供了call_id，在第一条消息前添加标识符
        if call_id is not None and messages and len(messages) > 0:
            messages[0]["content"] = f"[Call ID: {call_id}]\n\n" + messages[0]["content"]
            
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"LLM调用失败: {e}")
        return None

def get_diff_info(repo_name, commit_hash):
    """
    获取commit的diff信息
    """
    diff_path = f"{config.root_path}/knowledge_base/{repo_name}/modified_file/{commit_hash}/diff.txt"
    if os.path.exists(diff_path):
        with open(diff_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    return "Diff information not available"

def parse_code_examples(code_example_text):
    """
    尝试将LLM返回的代码示例文本解析为结构化的格式
    如果解析失败，则返回空列表
    """
    try:
        # 尝试直接解析为JSON
        if code_example_text.strip().startswith("[") and code_example_text.strip().endswith("]"):
            examples = json.loads(code_example_text)
            return examples if isinstance(examples, list) else []
        return []
    except:
        print("Failed to parse code examples as JSON, returning empty list")
        return []

def parse_application_conditions(conditions_text):
    """
    将应用条件文本转换为列表形式
    """
    # 尝试识别编号列表格式，如 "1. Condition" 或 "- Condition"
    conditions = []
    # 使用正则表达式匹配编号列表项
    pattern = r'(?:^|\n)(?:\d+\.|\-|\*)\s*(.*?)(?=(?:\n(?:\d+\.|\-|\*))|$)'
    matches = re.findall(pattern, conditions_text, re.DOTALL)
    if matches:
        conditions = [condition.strip() for condition in matches if condition.strip()]
    # 如果没有找到编号列表，则尝试按行分割
    if not conditions:
        conditions = [line.strip() for line in conditions_text.split('\n') if line.strip()]
    # 限制数量并确保每个条件是一句话
    conditions = conditions[:3]  # 最多保留3个条件
    # 确保每个条件是一句话（如果有多句，只保留第一句）
    for i in range(len(conditions)):
        sentences = re.split(r'(?<=[.!?])\s+', conditions[i])
        if sentences:
            conditions[i] = sentences[0]
    return conditions

def calculate_embeddings_and_similarities(summaries):
    """
    计算一组策略总结之间的相似度，并找出与其他总结平均相似度最高的那个
    """
    # 对每个总结计算嵌入向量
    embeddings = sentence_model.encode(summaries)
    # 计算每对总结之间的余弦相似度
    cos_sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    # 对于每个总结，计算它与所有其他总结的平均相似度
    avg_similarities = []
    for i in range(len(summaries)):
        # 从相似度矩阵中获取当前总结与所有总结的相似度
        similarities = cos_sim_matrix[i]
        # 移除自身相似度（值为1）
        other_similarities = np.delete(similarities, i)
        # 计算平均相似度
        avg_similarity = np.mean(other_similarities)
        avg_similarities.append(float(avg_similarity))  # 转换为Python内置float类型
    # 找出平均相似度最高的总结索引
    best_idx = int(np.argmax(avg_similarities))  # 转换为Python内置int类型
    return best_idx, avg_similarities[best_idx]

def build_context(commits):
    """
    根据配置为一组commits构建上下文信息
    """
    context = f"I'm analyzing a cluster of {len(commits)} commits that use similar performance optimization techniques. Here are the details of these commits: "
    for i, commit in enumerate(commits):
        context += f"\n\nCommit {i+1}:\n"
        # 添加 commit message (如果启用)
        if USE_COMMIT_MESSAGE:
            context += f"Message: {commit['message']}\n"
        # 添加优化总结
        context += f"Optimization summary: {commit['optimization_summary_final']}\n"
        # 添加 diff 信息 (如果启用)
        if USE_DIFF_INFO:
            diff_info = get_diff_info(commit['repository_name'], commit['hash'])
            context += f"Diff:\n{diff_info[:MAX_DIFF_LENGTH]}"
    return context

def generate_cluster_summary(cluster_id, cluster_data):
    """
    为每个聚类生成总结，使用self-consistency voting选取最佳结果
    """
    # 获取排序后的commits (按 authority_score 升序排列，越相似的放越后面)
    commits = cluster_data['commits']
    # 检查是否有 authority_score 字段，如果有就按它升序排序
    if commits and 'authority_score' in commits[0]:
        sorted_commits = sorted(commits, key=lambda c: c.get('authority_score', 0))
    else:
        sorted_commits = commits
    # 保存commit的总数和用于提示的commit总数
    total_commits = len(sorted_commits)
    commits_to_use = min(MAX_COMMITS_PER_CLUSTER, total_commits)
    # 统计被截断的diff数量
    truncated_diff_count = 0
    used_commits_count = 0
    # 准备要使用的commits
    commits_to_process = []
    commit_idx = 0
    while len(commits_to_process) < commits_to_use and commit_idx < total_commits:
        commit = sorted_commits[commit_idx]
        commit_idx += 1
        if USE_DIFF_INFO:
            diff_info = get_diff_info(commit['repository_name'], commit['hash'])
            # 检查diff是否需要截断
            is_truncated = len(diff_info) > MAX_DIFF_LENGTH
            # 如果设置跳过截断的diff，且当前diff需要截断，则跳过此commit
            if SKIP_TRUNCATED_DIFF and is_truncated and commit_idx < total_commits:
                print(f"Commit {commit['hash'][:8]} diff would be truncated, skipping")
                continue
            # 如果diff被截断，增加计数器
            if is_truncated:
                truncated_diff_count += 1
                print(f"Commit {commit['hash'][:8]} diff truncated from {len(diff_info)} to {MAX_DIFF_LENGTH} characters")
            # 保存优化后的diff信息
            commit['diff_info'] = diff_info[:MAX_DIFF_LENGTH]
        commits_to_process.append(commit)
        used_commits_count += 1
    # 输出使用的commits统计
    print(f"Cluster {cluster_id}: Using {used_commits_count}/{total_commits} commits, truncated diffs: {truncated_diff_count}")
    # 构建上下文信息
    context = build_context(commits_to_process)
    # 使用self-consistency voting，运行多次生成总结
    strategy_summaries = []
    code_examples_list = []
    application_conditions_list = []
    for i in range(CONSISTENCY_REPEATS):
        print(f"Cluster {cluster_id}: Running consistency repeat {i+1}/{CONSISTENCY_REPEATS}")
        # 创建当前迭代的唯一标识符
        iteration_id = f"cluster_{cluster_id}_iteration_{i+1}_of_{CONSISTENCY_REPEATS}"
        
        # 第一次对话：策略总结 (简短的一句话)
        messages = [
            {"role": "user", "content": context + "\n\nProvide a single, concise sentence that summarizes the common optimization strategy used across these commits. Be specific about the technical approach and focus on the core optimization technique."}
        ]
        # 使用标识符调用LLM
        strategy_summary = call_llm(messages, call_id=f"{iteration_id}_summary")
        strategy_summaries.append(strategy_summary)
        
        # 第二次对话：代码示例 (JSON格式的代码对)
        code_example_prompt = (
            "Generate exactly 2-3 before-and-after code examples that illustrate this optimization strategy. "
            "Try to extract real examples from the commit diffs if available, or create simplified examples that match the pattern. "
            "Format your response ONLY as a JSON array of pairs (nothing else), where each pair contains the code before and after optimization. "
            "Example format: [\n"
            "  [\"// Before\\ncode here\", \"// After\\ncode here\"],\n"
            "  [\"// Before\\nother example\", \"// After\\nother example\"]\n"
            "]\n"
            "Ensure the JSON is valid and properly formatted with no additional text before or after it."
        )
        messages = [
            {"role": "user", "content": context},
            {"role": "assistant", "content": strategy_summary},
            {"role": "user", "content": code_example_prompt}
        ]
        # 使用标识符调用LLM
        code_example_response = call_llm(messages, call_id=f"{iteration_id}_code")
        code_examples = parse_code_examples(code_example_response)
        code_examples_list.append(code_examples)
        
        # 第三次对话：应用条件 (简洁明确的条件)
        if USE_PROMPT:
            condition_prompt = (
                "List exactly 2-3 high-level conditions that would make code a good candidate for this optimization technique. "
                "Focus on architectural patterns, performance bottlenecks, or design characteristics rather than specific syntax details. "
                "Each condition should be expressed in a single sentence that captures a key insight for when this optimization is appropriate. "
                "Format your response as a numbered list with no explanations or additional text."
            )
        else:
            condition_prompt = (
                "List exactly 2-3 specific, testable conditions that code must satisfy to be a candidate for this optimization technique. "
                "Each condition must be expressed in a single sentence. "
                "Make each condition concrete and specific enough that it could potentially be checked by a static analysis tool. "
                "Format your response as a numbered list with no explanations or additional text."
            )
        messages = [
            {"role": "user", "content": context},
            {"role": "assistant", "content": strategy_summary},
            {"role": "user", "content": condition_prompt}
        ]
        # 使用标识符调用LLM
        application_conditions_response = call_llm(messages, call_id=f"{iteration_id}_conditions")
        application_conditions = parse_application_conditions(application_conditions_response)
        application_conditions_list.append(application_conditions)
    
    # 计算策略总结之间的相似度，选择最佳总结
    best_idx, best_similarity = calculate_embeddings_and_similarities(strategy_summaries)
    
    # 选择最佳结果
    best_strategy = strategy_summaries[best_idx]
    best_code_examples = code_examples_list[best_idx]
    best_conditions = application_conditions_list[best_idx]
    
    # 收集所有optimization_summary_final
    all_summaries = []
    for commit in commits:
        all_summaries.append(commit['optimization_summary_final'])
    
    return {
        'cluster_id': cluster_id,
        'size': cluster_data['size'],
        'used_commits_count': used_commits_count,
        'truncated_diff_count': truncated_diff_count if USE_DIFF_INFO else 0,
        'consistency_best_idx': best_idx,
        'consistency_best_similarity': float(best_similarity),  # 确保是Python原生类型
        'llm_summary': {
            'strategy_summary': best_strategy,
            'code_examples': best_code_examples,
            'application_conditions': best_conditions
        },
        'all_strategy_summaries': strategy_summaries,
        'all_optimization_summaries': all_summaries
    }

def main():
    # 加载输入文件
    print(f"Loading input file: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 第一部分：统计聚类信息
    print("Analyzing cluster sizes...")
    clusters_by_size = data.get('clusters_by_size', [])
    
    # 获取所有可能的commit_num值
    all_commit_nums = [c.get('commit_num', 0) for c in clusters_by_size]
    unique_commit_nums = sorted(set(all_commit_nums), reverse=True)
    
    # 计算每个阈值下的聚类数量
    cluster_count_by_threshold = {}
    for n in unique_commit_nums:
        count = sum(1 for c in clusters_by_size if c.get('commit_num', 0) >= n)
        cluster_count_by_threshold[n] = count
    
    # 第二部分：处理每个聚类的总结
    print(f"Generating summaries for clusters with {threshold}+ commits...")
    print(f"Using at most {MAX_COMMITS_PER_CLUSTER} commits per cluster, sorted by authority_score (ascending)")
    print(f"Self-consistency voting repeats: {CONSISTENCY_REPEATS}")
    print(f"Using commit message: {USE_COMMIT_MESSAGE}")
    print(f"Using new condition prompt: {USE_PROMPT}")
    if USE_DIFF_INFO:
        print(f"Using diff info: Yes (limit: {MAX_DIFF_LENGTH} characters)")
        print(f"Skip truncated diffs: {SKIP_TRUNCATED_DIFF}")
    else:
        print("Using diff info: No")
    
    clusters = data.get('clusters', {})
    filtered_clusters = {k: v for k, v in clusters.items() if v.get('size', 0) >= threshold}
    
    # 并行处理聚类总结
    cluster_summaries = []
    total_truncated_diffs = 0
    total_used_commits = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_cluster = {
            executor.submit(generate_cluster_summary, cluster_id, cluster_data):
            (cluster_id, cluster_data) for cluster_id, cluster_data in filtered_clusters.items()
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_cluster),
                          total=len(filtered_clusters),
                          desc="Processing clusters"):
            try:
                result = future.result()
                if result:
                    cluster_summaries.append(result)
                    total_truncated_diffs += result.get('truncated_diff_count', 0)
                    total_used_commits += result.get('used_commits_count', 0)
            except Exception as e:
                cluster_id = future_to_cluster[future][0]
                print(f"Error processing cluster {cluster_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # 按聚类大小排序结果
    cluster_summaries.sort(key=lambda x: x.get('size', 0), reverse=True)
    
    # 构建输出结果
    result = {
        'cluster_count_by_threshold': cluster_count_by_threshold,
        'cluster_summaries': cluster_summaries,
        'metadata': {
            'use_diff_info': USE_DIFF_INFO,
            'use_commit_message': USE_COMMIT_MESSAGE,
            'max_diff_length': MAX_DIFF_LENGTH if USE_DIFF_INFO else None,
            'skip_truncated_diff': SKIP_TRUNCATED_DIFF if USE_DIFF_INFO else None,
            'max_commits_per_cluster': MAX_COMMITS_PER_CLUSTER,
            'consistency_repeats': CONSISTENCY_REPEATS,
            'USE_PROMPT': USE_PROMPT,
            'threshold': threshold,
            'total_clusters_analyzed': len(cluster_summaries),
            'total_used_commits': total_used_commits,
            'total_truncated_diffs': total_truncated_diffs
        }
    }
    
    # 保存输出文件
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Total clusters analyzed: {len(cluster_summaries)}")
    print(f"Total commits used: {total_used_commits}")
    if USE_DIFF_INFO:
        percentage = (total_truncated_diffs / max(1, total_used_commits)) * 100
        print(f"Total truncated diffs: {total_truncated_diffs} ({percentage:.1f}% of used commits)")
    print(f"\nSaving results to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # 使用自定义编码器处理numpy类型
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print("Analysis completed!")

if __name__ == "__main__":
    main()