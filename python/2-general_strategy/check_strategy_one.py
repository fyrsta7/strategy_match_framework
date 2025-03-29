import os
import json
import sys
import re
import concurrent.futures
from tqdm import tqdm
from typing import Dict, List, Any
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 每个commit评估的重复次数
N_REPEATED_EVALUATIONS = 5
# 是否强制重新生成所有评估结果
REGENERATE_IF_EXISTS = True
# 并行处理的最大线程数
MAX_WORKERS = 16
# 最大重试次数
MAX_RETRY_ATTEMPTS = 3
# 所用LLM模型
MODEL = "dsv3"
MODEL_NAME = config.xmcp_deepseek_model if MODEL == "dsv3" else config.xmcp_qwen_model
# 输入文件：包含聚类和对应commit信息的合并数据文件
FILE_DIR = config.root_path + "python/2-general_strategy/result/kmeans_40000_dsv3_true_14000/"
INPUT_FILE_NAME = "order_sum_11_100_10_dsv3_full_16"
INPUT_COMBINED_CLUSTERS_FILE = FILE_DIR + f"{INPUT_FILE_NAME}.json"
# 输出文件：包含每个commit是否可应用所在聚类策略的评估结果
OUTPUT_ASSESSMENT_FILE = FILE_DIR + f"{INPUT_FILE_NAME}_appone_{MODEL}.json"

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
def assess_strategy_applicability_single(client, commit: Dict[str, Any], strategy_summary: str, application_conditions: List[str], attempt_number: int) -> Dict[str, Any]:
    """评估一个commit的代码是否可以应用聚类中的优化策略（单次评估）"""
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
    
    # 使用尝试次数作为标识符，确保每次请求唯一
    attempt_id = f"attempt_{attempt_number}"
    
    # 构建评估提示
    prompt = (
        f"[{attempt_id}] I need to determine if a specific optimization strategy can be applied to the following C/C++ code. "
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
                print(f"重试评估 {commit_hash} (重试 {retry_count+1}/{MAX_RETRY_ATTEMPTS}): {e}")
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
def assess_strategy_applicability(client, commit: Dict[str, Any], strategy_summary: str, application_conditions: List[str]) -> Dict[str, Any]:
    """评估一个commit的代码是否可以应用聚类中的优化策略（多次评估并投票）"""
    repo_name = commit.get('repository_name', '')
    commit_hash = commit.get('hash', '')
    
    # 执行N次重复评估
    assessments = []
    for i in range(N_REPEATED_EVALUATIONS):
        # 将尝试次数(i+1)传递给评估函数
        result = assess_strategy_applicability_single(client, commit, strategy_summary, application_conditions, i+1)
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
def process_commit(args):
    """用于并行处理的包装函数"""
    client, commit, strategy_summary, application_conditions = args
    return assess_strategy_applicability(client, commit, strategy_summary, application_conditions)
def main():
    # 加载合并后的聚类数据
    print(f"正在加载聚类数据: {INPUT_COMBINED_CLUSTERS_FILE}")
    try:
        with open(INPUT_COMBINED_CLUSTERS_FILE, 'r', encoding='utf-8') as f:
            combined_data = json.load(f)
    except Exception as e:
        print(f"错误：无法读取聚类数据文件 {INPUT_COMBINED_CLUSTERS_FILE}。{e}")
        sys.exit(1)
    
    # 检查是否已存在评估结果文件
    existing_cluster_assessments = {}
    if os.path.exists(OUTPUT_ASSESSMENT_FILE) and not REGENERATE_IF_EXISTS:
        try:
            with open(OUTPUT_ASSESSMENT_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                # 创建一个哈希映射以快速查找已评估的commit
                existing_cluster_assessments = existing_results.get("cluster_assessments", {})
                
                # 创建一个映射，将commit_hash映射到其评估结果
                existing_commit_assessments = {}
                for cluster_id, assessments in existing_cluster_assessments.items():
                    for assessment in assessments:
                        existing_commit_assessments[assessment["commit_hash"]] = assessment
                        
        except Exception as e:
            existing_cluster_assessments = {}
            print(f"无法读取现有评估文件，将创建新文件: {e}")
    else:
        existing_cluster_assessments = {}
    
    # 初始化OpenAI客户端
    client = get_client()
    
    # 为新的输出结构准备数据
    cluster_assessments = {}
    all_tasks = []
    commit_to_cluster = {}  # 映射commit到其所属的cluster_id
    
    # 遍历每个聚类
    for cluster in combined_data.get("combined_clusters", []):
        cluster_id = cluster.get("cluster_id")
        strategy_summary = cluster.get("summary", {}).get("strategy_summary", "")
        application_conditions = cluster.get("summary", {}).get("application_conditions", [])
        
        # 初始化该聚类的评估结果列表
        if cluster_id not in cluster_assessments:
            # 如果有现有结果且不需要重生成，则沿用
            if cluster_id in existing_cluster_assessments and not REGENERATE_IF_EXISTS:
                cluster_assessments[cluster_id] = existing_cluster_assessments[cluster_id]
            else:
                cluster_assessments[cluster_id] = []
        
        # 处理每个聚类中的所有commit
        for commit in cluster.get("commits", []):
            commit_hash = commit.get("hash")
            commit_to_cluster[commit_hash] = cluster_id
            
            # 如果已有评估结果且不需要重新生成，则跳过
            if (cluster_id in existing_cluster_assessments and 
                any(assessment["commit_hash"] == commit_hash for assessment in existing_cluster_assessments[cluster_id]) and
                not REGENERATE_IF_EXISTS):
                continue
            
            # 添加到并行处理队列
            all_tasks.append((client, commit, strategy_summary, application_conditions))
    
    # 并行处理所有需要评估的commit
    if all_tasks:
        print(f"需要评估 {len(all_tasks)} 个提交...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(
                executor.map(process_commit, all_tasks),
                total=len(all_tasks),
                desc="评估提交"
            ))
        
        # 按聚类组织评估结果
        for result in results:
            commit_hash = result["commit_hash"]
            cluster_id = commit_to_cluster[commit_hash]
            
            if cluster_id not in cluster_assessments:
                cluster_assessments[cluster_id] = []
            
            cluster_assessments[cluster_id].append(result)
    else:
        print("所有提交已评估，无需更新")
    
    # 计算各聚类的应用性统计信息
    cluster_statistics = {}
    total_applicable_count = 0
    total_assessments = 0
    
    for cluster_id, assessments in cluster_assessments.items():
        applicable_count = sum(1 for a in assessments if a.get("voting_result", False))
        cluster_statistics[cluster_id] = {
            "total_commits": len(assessments),
            "applicable_commits": applicable_count,
            "applicable_percentage": applicable_count / len(assessments) if assessments else 0
        }
        total_applicable_count += applicable_count
        total_assessments += len(assessments)
    
    # 生成最终输出
    output_data = {
        "metadata": {
            "model": MODEL,
            "total_assessments": total_assessments,
            "applicable_count": total_applicable_count,
            "repeated_evaluations": N_REPEATED_EVALUATIONS,
            "generated_at": __import__('datetime').datetime.now().isoformat()
        },
        "cluster_statistics": cluster_statistics,
        "cluster_assessments": cluster_assessments
    }
    
    # 保存结果
    print(f"保存评估结果到: {OUTPUT_ASSESSMENT_FILE}")
    with open(OUTPUT_ASSESSMENT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"完成！共评估了 {total_assessments} 个提交，其中 {total_applicable_count} 个可应用聚类策略")
if __name__ == "__main__":
    main()