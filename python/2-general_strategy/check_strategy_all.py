import os
import json
import sys
import re
import uuid
import concurrent.futures
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI
import datetime

# 每个commit评估的重复次数
N_REPEATED_EVALUATIONS = 5
# 是否强制重新生成所有评估结果
REGENERATE_IF_EXISTS = False
# 并行处理的最大线程数
MAX_WORKERS = 16
# 最大重试次数
MAX_RETRY_ATTEMPTS = 3
# 所用LLM模型
MODEL = "dsv3"
MODEL_NAME = config.xmcp_deepseek_model if MODEL == "dsv3" else config.xmcp_qwen_model
# 输入文件：包含聚类和对应commit信息的合并数据文件
FILE_DIR = config.root_path + "python/2-general_strategy/result/kmeans_10000_dsv3_true_3000/"
INPUT_FILE_NAME = "order_sum_16_100_dsv3_full_16"
INPUT_COMBINED_CLUSTERS_FILE = FILE_DIR + f"{INPUT_FILE_NAME}.json"
# 输出文件：包含每个commit是否可应用所有聚类策略的评估结果
OUTPUT_ASSESSMENT_FILE = FILE_DIR + f"{INPUT_FILE_NAME}_appall_{MODEL}.json"

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

def get_function_code(repo_name: str, commit_hash: str) -> str:
    """获取commit对应的函数修改前代码"""
    file_path = os.path.join(
        config.root_path,
        'knowledge_base',
        repo_name,
        'modified_file',
        commit_hash,
        'before_func.txt'
    )
    
    if not os.path.exists(file_path):
        return f"ERROR: File not found at {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"ERROR: Failed to read file {file_path}. {e}"

def assess_strategy_applicability_single(client, commit: Dict[str, Any], cluster_id: str, strategy_summary: str, 
                                      application_conditions: List[str], attempt_number: int) -> Dict[str, Any]:
    """评估一个commit的代码是否可以应用特定聚类中的优化策略（单次评估）"""
    repo_name = commit.get('repository_name', '')
    commit_hash = commit.get('hash', '')
    
    # 获取函数代码
    function_code = get_function_code(repo_name, commit_hash)
    if function_code.startswith("ERROR:"):
        return {
            "applicable": False,
            "explanation": function_code,
            "error": function_code
        }
    
    # 使用唯一标识符，确保每次请求唯一
    # 组合尝试次数和UUID以确保唯一性
    unique_id = f"attempt_{attempt_number}_cluster_{cluster_id}_{str(uuid.uuid4())[:8]}"
    
    # 构建评估提示
    prompt = (
        f"[{unique_id}] I need to determine if a specific optimization strategy can be applied to the following C/C++ code. "
        f"\n\n# Optimization Strategy:\n{strategy_summary}"
        f"\n\n# Application Conditions:"
    )
    
    # 添加应用条件
    for i, condition in enumerate(application_conditions, 1):
        prompt += f"\n{i}. {condition}"
    
    prompt += (
        f"\n\n# Code to Analyze:\n```cpp\n{function_code}\n```"
        f"\n\nBased on the optimization strategy and application conditions, is this optimization applicable to the given code? "
        f"Provide your answer in JSON format with the following structure:"
        f"\n```json\n{{"
        f"\n    \"applicable\": true/false, // Whether the optimization can be applied"
        f"\n    \"explanation\": \"2-3 sentences explaining your reasoning\""
        f"\n}}\n```"
        f"\nFocus on whether the code exhibits the patterns that this optimization strategy is designed to address."
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    # 尝试调用LLM，必要时重试
    for retry_count in range(MAX_RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # 提取JSON格式的回答
            try:
                # 尝试从可能包含其他内容的响应中提取JSON
                json_match = re.search(r'```json\s*({.*?})\s*```', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # 尝试直接解析整个响应
                    json_str = response_content
                
                result = json.loads(json_str)
                
                # 确保结果包含所需的键
                if 'applicable' not in result or 'explanation' not in result:
                    raise ValueError("Response missing required fields")
                
                return result
                
            except Exception as json_err:
                # 如果无法解析JSON，尝试通过文本分析提取信息
                applicable = "true" in response_content.lower() and "applicable" in response_content.lower()
                
                # 提取解释（简单处理）
                explanation = response_content
                if len(explanation) > 500:
                    explanation = explanation[:497] + "..."
                
                return {
                    "applicable": applicable,
                    "explanation": explanation,
                    "parse_error": str(json_err)
                }
            
        except Exception as e:
            if retry_count < MAX_RETRY_ATTEMPTS - 1:
                continue
            
            return {
                "applicable": False,
                "explanation": f"Failed after {MAX_RETRY_ATTEMPTS} attempts: {e}",
                "error": str(e)
            }
    
    return {
        "applicable": False,
        "explanation": "Failed to get response after maximum retries",
        "error": "Max retries exceeded"
    }

def assess_strategy_applicability(client, commit: Dict[str, Any], cluster_id: str, strategy_summary: str, 
                               application_conditions: List[str]) -> Dict[str, Any]:
    """评估一个commit的代码是否可以应用特定聚类中的优化策略（多次评估并投票）"""
    repo_name = commit.get('repository_name', '')
    commit_hash = commit.get('hash', '')
    
    # 执行N次重复评估
    assessments = []
    for i in range(N_REPEATED_EVALUATIONS):
        result = assess_strategy_applicability_single(
            client, commit, cluster_id, strategy_summary, application_conditions, i+1
        )
        assessments.append(result)
    
    # 进行self-consistency voting
    applicable_votes = [assessment["applicable"] for assessment in assessments]
    vote_counter = Counter(applicable_votes)
    majority_vote = vote_counter.most_common(1)[0][0]
    vote_confidence = vote_counter[majority_vote] / len(applicable_votes)
    
    # 构建最终结果
    return {
        "commit_hash": commit_hash,
        "repository": repo_name,
        "individual_assessments": assessments,
        "voting_result": majority_vote,
        "voting_confidence": vote_confidence,
        "majority_explanation": next((a["explanation"] for a in assessments if a["applicable"] == majority_vote), "No explanation found")
    }

def process_task(args):
    """用于并行处理的包装函数"""
    client, commit, cluster_id, strategy_summary, application_conditions = args
    try:
        result = assess_strategy_applicability(client, commit, cluster_id, strategy_summary, application_conditions)
        return commit.get("hash"), cluster_id, result
    except Exception as e:
        print(f"处理任务时发生错误: {e}")
        return commit.get("hash"), cluster_id, {
            "commit_hash": commit.get("hash"),
            "repository": commit.get("repository_name", ""),
            "error": str(e),
            "voting_result": False,
            "voting_confidence": 0,
            "majority_explanation": f"Error during evaluation: {e}",
            "individual_assessments": []
        }

def main():
    # 加载合并后的聚类数据
    print(f"正在加载聚类数据: {INPUT_COMBINED_CLUSTERS_FILE}")
    try:
        with open(INPUT_COMBINED_CLUSTERS_FILE, 'r', encoding='utf-8') as f:
            combined_data = json.load(f)
    except Exception as e:
        print(f"错误：无法读取聚类数据文件 {INPUT_COMBINED_CLUSTERS_FILE}。{e}")
        sys.exit(1)
    
    # 提取所有聚类的策略信息
    clusters_info = {}
    all_commits = set()
    for cluster in combined_data.get("combined_clusters", []):
        cluster_id = cluster.get("cluster_id")
        strategy_summary = cluster.get("summary", {}).get("strategy_summary", "")
        application_conditions = cluster.get("summary", {}).get("application_conditions", [])
        
        clusters_info[cluster_id] = {
            "strategy_summary": strategy_summary,
            "application_conditions": application_conditions
        }
        
        # 收集所有唯一的commits
        for commit in cluster.get("commits", []):
            all_commits.add(commit.get("hash"))
    
    # 创建所有commit的映射，以便轻松访问
    commit_map = {}
    for cluster in combined_data.get("combined_clusters", []):
        for commit in cluster.get("commits", []):
            commit_hash = commit.get("hash")
            commit_map[commit_hash] = commit
    
    # 检查是否已存在评估结果文件
    existing_results = {}
    if os.path.exists(OUTPUT_ASSESSMENT_FILE) and not REGENERATE_IF_EXISTS:
        try:
            with open(OUTPUT_ASSESSMENT_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except Exception as e:
            existing_results = {}
            print(f"无法读取现有评估文件，将创建新文件: {e}")
    
    # 初始化OpenAI客户端
    client = get_client()
    
    # 准备所有需要评估的任务
    all_tasks = []
    commit_to_original_cluster = {}  # 记录commit原始所属的聚类
    
    # 为每个commit的每个聚类策略准备评估任务
    for cluster in combined_data.get("combined_clusters", []):
        cluster_id = cluster.get("cluster_id")
        
        for commit in cluster.get("commits", []):
            commit_hash = commit.get("hash")
            # 记录commit原始所属的聚类
            commit_to_original_cluster[commit_hash] = cluster_id
    
    # 初始化结果字典 - 按聚类组织
    cluster_assessments = defaultdict(dict)
    
    # 如果有现有结果且不需要重新生成，则加载
    if not REGENERATE_IF_EXISTS and "cluster_assessments" in existing_results:
        cluster_assessments = defaultdict(dict, existing_results["cluster_assessments"])
    
    # 对每个commit评估每个聚类的策略
    for commit_hash, commit in commit_map.items():
        # 对每个聚类的策略进行评估
        for cluster_id, info in clusters_info.items():
            # 检查是否已有该commit在该聚类的评估结果
            if (not REGENERATE_IF_EXISTS and 
                cluster_id in cluster_assessments and
                commit_hash in cluster_assessments[cluster_id]):
                continue
                
            # 添加到并行处理队列
            all_tasks.append((
                client, 
                commit, 
                cluster_id, 
                info["strategy_summary"], 
                info["application_conditions"]
            ))
    
    # 并行处理所有需要评估的任务
    if all_tasks:
        print(f"需要评估 {len(all_tasks)} 个任务 (涉及 {len(set(task[1]['hash'] for task in all_tasks))} 个提交)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_iter = executor.map(process_task, all_tasks)
            
            # 使用tqdm显示进度
            total_tasks = len(all_tasks)
            completed_tasks = 0
            
            with tqdm(total=total_tasks, desc="评估策略应用性") as pbar:
                for commit_hash, cluster_id, result in results_iter:
                    # 将结果保存到对应的聚类下
                    cluster_assessments[cluster_id][commit_hash] = result
                    completed_tasks += 1
                    pbar.update(1)
                    
                    # 定期保存结果，以防中断
                    if completed_tasks % 50 == 0:
                        interim_results = {
                            "metadata": {
                                "model": MODEL,
                                "repeated_evaluations": N_REPEATED_EVALUATIONS,
                                "generated_at": datetime.datetime.now().isoformat(),
                                "completed_tasks": completed_tasks,
                                "total_tasks": total_tasks,
                                "progress": f"{completed_tasks}/{total_tasks}"
                            },
                            "cluster_assessments": dict(cluster_assessments)
                        }
                        with open(OUTPUT_ASSESSMENT_FILE, 'w', encoding='utf-8') as f:
                            json.dump(interim_results, f, indent=2, ensure_ascii=False)
    else:
        print("所有提交已评估，无需更新")
    
    # 计算统计数据
    total_commits = len(commit_map)
    commits_applicable_to_original_cluster = 0
    commits_applicable_to_any_cluster = 0
    total_applicable_strategies = 0
    applicable_strategies_by_commit = defaultdict(int)
    applicable_strategies_by_cluster = defaultdict(list)
    commits_in_cluster = defaultdict(set)

    # 统计每个聚类包含的提交数量
    for commit_hash, cluster_id in commit_to_original_cluster.items():
        commits_in_cluster[cluster_id].add(commit_hash)
    
    # 分析结果
    for cluster_id, commit_assessments in cluster_assessments.items():
        for commit_hash, assessment in commit_assessments.items():
            # 如果该策略适用于该commit
            if assessment.get("voting_result", False):
                # 记录策略适用情况
                applicable_strategies_by_cluster[cluster_id].append(commit_hash)
                applicable_strategies_by_commit[commit_hash] += 1
                total_applicable_strategies += 1
                
                # 检查是否是commit的原始聚类
                if commit_to_original_cluster.get(commit_hash) == cluster_id:
                    commits_applicable_to_original_cluster += 1
    
    # 计算可应用任何策略的commit数量
    commits_applicable_to_any = set()
    for commit_hash in commit_map.keys():  # 使用所有commit作为基础
        if applicable_strategies_by_commit.get(commit_hash, 0) > 0:
            commits_applicable_to_any.add(commit_hash)
    commits_applicable_to_any_cluster = len(commits_applicable_to_any)
        
    # 计算平均每个commit可应用的策略数量
    avg_applicable_strategies_per_commit = total_applicable_strategies / total_commits if total_commits > 0 else 0
    
    # 生成统计摘要数据
    statistics = {
        "total_commits": total_commits,
        "total_clusters": len(clusters_info),
        "commits_applicable_to_original_cluster": commits_applicable_to_original_cluster,
        "commits_applicable_to_original_cluster_percentage": 
            commits_applicable_to_original_cluster / total_commits if total_commits > 0 else 0,
        "commits_applicable_to_any_cluster": commits_applicable_to_any_cluster,
        "commits_applicable_to_any_cluster_percentage": 
            commits_applicable_to_any_cluster / total_commits if total_commits > 0 else 0,
        "total_applicable_strategies": total_applicable_strategies,
        "avg_applicable_strategies_per_commit": avg_applicable_strategies_per_commit,
        "cluster_applicability": {
            cluster_id: {
                "applicable_count": len(applicable_strategies_by_cluster[cluster_id]),
                "percentage": len(applicable_strategies_by_cluster[cluster_id]) / total_commits if total_commits > 0 else 0,
                "total_commits_in_cluster": len(commits_in_cluster[cluster_id]),
                "applicable_commits_from_cluster": len([c for c in applicable_strategies_by_cluster[cluster_id] if c in commits_in_cluster[cluster_id]]),
                "applicable_commits_from_other_clusters": len([c for c in applicable_strategies_by_cluster[cluster_id] if c not in commits_in_cluster[cluster_id]])
            }
            for cluster_id in clusters_info.keys()
        },
        "commit_applicability_distribution": {
            str(count): len([c for c, app_count in applicable_strategies_by_commit.items() if app_count == count])
            for count in range(max(applicable_strategies_by_commit.values()) + 1 if applicable_strategies_by_commit else 1)
        }
    }
    
    # 生成最终输出
    output_data = {
        "metadata": {
            "model": MODEL,
            "repeated_evaluations": N_REPEATED_EVALUATIONS,
            "generated_at": datetime.datetime.now().isoformat(),
            "input_file": INPUT_COMBINED_CLUSTERS_FILE
        },
        "statistics": statistics,
        "cluster_assessments": dict(cluster_assessments)
    }
    
    # 保存结果
    print(f"保存评估结果到: {OUTPUT_ASSESSMENT_FILE}")
    with open(OUTPUT_ASSESSMENT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 输出统计概要
    print("\n--- 评估统计概要 ---")
    print(f"总commit数: {total_commits}")
    print(f"总策略数: {len(clusters_info)}")
    print(f"可应用原始聚类策略的commit数: {commits_applicable_to_original_cluster} ({commits_applicable_to_original_cluster/total_commits*100:.2f}%)")
    print(f"可应用任何策略的commit数: {commits_applicable_to_any_cluster} ({commits_applicable_to_any_cluster/total_commits*100:.2f}%)")
    print(f"平均每个commit可应用策略数: {avg_applicable_strategies_per_commit:.2f}")
    print("------------------------")

if __name__ == "__main__":
    main()