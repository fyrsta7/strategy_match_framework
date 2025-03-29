import os
import json
import sys
import re
import random
import concurrent.futures
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 全局变量
# 输入文件：包含聚类和评估结果的文件
INPUT_ASSESSMENT_FILE = config.root_path + "python/2-general_strategy/result/kmeans_40000_dsv3_true_14000/order_sum_11_100_10_dsv3_full_16_appone_qwenmax.json"
COMMIT_RESULT_DIR = config.root_path + "python/2-general_strategy/result/kmeans_40000_dsv3_true_14000/commit_result/order_sum_11_100_10_dsv3_full_16_appone_qwenmax/"
# 是否强制重新生成所有优化结果
REGENERATE_IF_EXISTS = False
# 是否沿用已经存在的结果，只生成缺失的结果
REUSE_EXISTING_RESULTS = True
# 每个函数优化的次数
OPTIMIZATION_ATTEMPTS = 3
# 最大重试次数
MAX_RETRY_ATTEMPTS = 3
# 所用LLM模型
MODEL_NAME = "qwen-max"
MODEL_API_NAME = config.xmcp_qwen_model if MODEL_NAME == "qwen-max" else config.xmcp_deepseek_model
TEMPERATURE = 0.1
# 并行处理的最大线程数
MAX_WORKERS = 32
# 每个聚类中随机选择的参考commit数量
REFERENCE_COMMIT_COUNT = 3

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

def extract_code_blocks(text):
    """从文本中提取所有代码块"""
    pattern = r"```(?:[a-zA-Z0-9+]*\n)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def get_function_code(repo_name, commit_hash):
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
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return None

def get_diff_info(repo_name, commit_hash):
    """获取commit的diff信息"""
    diff_path = os.path.join(
        config.root_path,
        'knowledge_base',
        repo_name,
        'modified_file',
        commit_hash,
        'diff.txt'
    )
    
    if os.path.exists(diff_path):
        try:
            with open(diff_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            pass
    
    return "Diff information not available"

def get_cluster_summary(cluster_id, assessment_data):
    """从聚类评估数据中获取该聚类的优化策略摘要"""
    # 从第一个commit的majority_explanation中提取策略信息
    if str(cluster_id) in assessment_data["cluster_assessments"]:
        commits = assessment_data["cluster_assessments"][str(cluster_id)]
        if commits and len(commits) > 0:
            return commits[0].get("majority_explanation", "No strategy summary available")
    return "No strategy summary available"

def check_existing_results(repo_name, commit_hash):
    """检查commit的优化结果是否已存在，返回缺失的尝试编号"""
    result_dir = os.path.join(
        COMMIT_RESULT_DIR,
        repo_name,
        commit_hash,
        MODEL_NAME
    )
    
    missing_attempts = []
    for attempt in range(1, OPTIMIZATION_ATTEMPTS + 1):
        result_file_path = os.path.join(result_dir, f"{attempt}.txt")
        if not os.path.exists(result_file_path):
            missing_attempts.append(attempt)
    
    return missing_attempts

def optimize_function_single_attempt(client, target_commit, cluster_id, strategy_summary, reference_commits, attempt_number):
    """单次调用LLM优化函数"""
    repo_name = target_commit.get("repository", "")
    commit_hash = target_commit.get("commit_hash", "")
    
    # 构建新的结果存储路径
    result_dir = os.path.join(
        COMMIT_RESULT_DIR,
        repo_name,
        commit_hash,
        MODEL_NAME
    )
    
    # 确保目录存在
    os.makedirs(result_dir, exist_ok=True)
    
    result_file_path = os.path.join(result_dir, f"{attempt_number}.txt")
    
    # 跳过已存在的结果（除非允许重生成）
    if os.path.exists(result_file_path) and not REGENERATE_IF_EXISTS:
        return f"跳过提交 {commit_hash} 的第 {attempt_number} 次优化，因为结果已存在。"
    
    # 获取目标函数代码
    target_function_code = get_function_code(repo_name, commit_hash)
    if not target_function_code:
        return f"错误：无法获取提交 {commit_hash} 的函数代码。"
    
    # 收集参考commit的diff信息
    reference_examples = []
    for ref_commit in reference_commits:
        ref_repo = ref_commit.get("repository", "")
        ref_hash = ref_commit.get("commit_hash", "")
        ref_message = ""  # 这个字段在新格式中可能没有，所以使用空值
        
        diff_info = get_diff_info(ref_repo, ref_hash)
        if diff_info and diff_info != "Diff information not available":
            # 将diff信息截断，保持在合理长度
            if len(diff_info) > 2000:
                diff_info = diff_info[:2000] + "...[truncated]"
            
            reference_examples.append({
                "message": ref_message,
                "diff": diff_info
            })
    
    # 添加标识符以防止缓存命中
    prompt_id = f"[Commit: {commit_hash}, Attempt: {attempt_number} of {OPTIMIZATION_ATTEMPTS}]\n\n"
    
    # 构建优化提示
    prompt = (
        prompt_id +  # 在提示的开头添加标识符
        "You are a performance optimization expert for C/C++ code. "
        "I need you to optimize a function using a specific performance optimization strategy.\n\n"
        
        f"## Optimization Strategy\n{strategy_summary}\n\n"
        
        "## Reference Examples\n"
        "Here are some examples of commits that implemented this optimization strategy:\n\n"
    )
    
    # 添加参考示例
    for i, example in enumerate(reference_examples, 1):
        prompt += f"### Example {i}\n"
        prompt += f"Commit message: {example['message']}\n"
        prompt += f"Diff:\n```\n{example['diff']}\n```\n\n"
    
    prompt += (
        "## Your Task\n"
        "Please optimize the following C/C++ function using the strategy described above. "
        "Ensure that the semantics of the function remain unchanged and the performance is improved. "
        "Only modify code that needs optimization according to the strategy. "
        "Do not change any code style, formatting, or comments unless necessary for the optimization. "
        "You can explain your optimization approach, but make sure to include exactly ONE code block "
        "containing the complete optimized function. "
        f"Here is the function to optimize:\n\n```cpp\n{target_function_code}\n```\n\n"
        "Provide your complete optimized function within a single markdown code block using triple backticks (```)."
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    # 尝试调用LLM，必要时重试
    for retry_count in range(MAX_RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model = MODEL_API_NAME,
                messages = messages,
                temperature = TEMPERATURE
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # 提取代码块
            code_blocks = extract_code_blocks(response_content)
            
            # 检查代码块数量
            if len(code_blocks) == 0:
                if retry_count < MAX_RETRY_ATTEMPTS - 1:
                    # 添加重试标识符
                    retry_prompt = f"[Commit: {commit_hash}, Attempt: {attempt_number}, Retry: {retry_count + 1}]\n\n"
                    messages.append({"role": "assistant", "content": response_content})
                    messages.append({
                        "role": "user", 
                        "content": retry_prompt + "You didn't include a code block with your optimized function. Please provide your complete optimized function in a single markdown code block using triple backticks (```)."
                    })
                    continue
                else:
                    return f"错误：提交 {commit_hash} 的第 {attempt_number} 次优化经过 {MAX_RETRY_ATTEMPTS} 次尝试后仍未生成代码块"
            
            elif len(code_blocks) > 1:
                if retry_count < MAX_RETRY_ATTEMPTS - 1:
                    # 添加重试标识符
                    retry_prompt = f"[Commit: {commit_hash}, Attempt: {attempt_number}, Retry: {retry_count + 1}]\n\n"
                    messages.append({"role": "assistant", "content": response_content})
                    messages.append({
                        "role": "user", 
                        "content": retry_prompt + "You included multiple code blocks. Please provide only ONE code block with your complete optimized function."
                    })
                    continue
                else:
                    return f"错误：提交 {commit_hash} 的第 {attempt_number} 次优化经过 {MAX_RETRY_ATTEMPTS} 次尝试后仍生成了多个代码块"
            
            # 如果只有一个代码块，则保存结果
            optimized_function = code_blocks[0]
            
            try:
                with open(result_file_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_function)
                return f"提交 {commit_hash}: 第 {attempt_number} 次聚类优化后的函数已保存到 {result_file_path}"
            except Exception as e:
                return f"错误：无法写入文件 {result_file_path}。{e}"
        
        except Exception as e:
            if retry_count < MAX_RETRY_ATTEMPTS - 1:
                continue
            return f"请求 OpenAI 时出错（Commit: {commit_hash}，尝试: {attempt_number}，重试: {retry_count + 1}）：{e}"
    
    return f"错误：提交 {commit_hash} 的第 {attempt_number} 次优化全部失败"

def optimize_function(client, target_commit, cluster_id, strategy_summary, reference_commits):
    """对函数进行多次优化，每次产生一个单独的结果文件"""
    results = []
    repo_name = target_commit.get("repository", "")
    commit_hash = target_commit.get("commit_hash", "")
    
    # 如果启用了复用现有结果，先检查已有结果，只生成缺失的部分
    if REUSE_EXISTING_RESULTS:
        attempts_to_process = check_existing_results(repo_name, commit_hash)
        if len(attempts_to_process) == 0:
            return [f"提交 {commit_hash}: 所有优化结果已存在，无需重新生成"]
        
        print(f"提交 {commit_hash}: 需要生成的尝试编号: {attempts_to_process}")
        
        for attempt in attempts_to_process:
            result = optimize_function_single_attempt(
                client, target_commit, cluster_id, strategy_summary, reference_commits, attempt
            )
            results.append(result)
    else:
        # 否则按照常规方式处理所有尝试
        for attempt in range(1, OPTIMIZATION_ATTEMPTS + 1):
            result = optimize_function_single_attempt(
                client, target_commit, cluster_id, strategy_summary, reference_commits, attempt
            )
            results.append(result)
    
    return results

def process_item(args):
    """用于并行处理的包装函数"""
    client, commit, cluster_id, strategy_summary, reference_commits = args
    return optimize_function(client, commit, cluster_id, strategy_summary, reference_commits)

def main():
    # 加载评估结果数据
    print(f"正在加载评估数据: {INPUT_ASSESSMENT_FILE}")
    try:
        with open(INPUT_ASSESSMENT_FILE, 'r', encoding='utf-8') as f:
            assessment_data = json.load(f)
    except Exception as e:
        print(f"错误：无法读取评估数据文件 {INPUT_ASSESSMENT_FILE}。{e}")
        sys.exit(1)
    
    # 准备需要处理的commit和对应的聚类数据
    items_to_process = []
    client = get_client()
    
    # 遍历所有聚类
    for cluster_id, commits in assessment_data.get("cluster_assessments", {}).items():
        print(f"处理聚类 {cluster_id}，包含 {len(commits)} 个commit")
        
        # 筛选出voting_result为true的commit
        applicable_commits = [commit for commit in commits if commit.get("voting_result", False)]
        print(f"  - 其中 {len(applicable_commits)} 个commit被评估为适用该策略")
        
        if not applicable_commits:
            continue
        
        # 获取聚类的优化策略摘要
        strategy_summary = get_cluster_summary(cluster_id, assessment_data)
        
        # 从适用的commit中随机选择参考commit
        reference_commits = []
        if len(applicable_commits) > 1:  # 确保至少有2个commit才执行随机选择
            # 计算要选择的参考commit数量（不包括当前要处理的commit）
            sample_size = min(REFERENCE_COMMIT_COUNT, len(applicable_commits) - 1)
            if sample_size > 0:
                reference_commits = random.sample(applicable_commits, sample_size)
        
        # 为每个适用的commit创建优化任务
        for commit in applicable_commits:
            # 避免使用commit自身作为参考
            commit_references = [ref for ref in reference_commits if ref.get("commit_hash") != commit.get("commit_hash")]
            
            # 如果参考commit数量不足，尝试再随机选择一些
            if len(commit_references) < REFERENCE_COMMIT_COUNT and len(applicable_commits) > 1:
                additional_refs = [c for c in applicable_commits if c.get("commit_hash") != commit.get("commit_hash") 
                                 and c not in commit_references]
                
                if additional_refs:
                    additional_count = min(REFERENCE_COMMIT_COUNT - len(commit_references), len(additional_refs))
                    commit_references.extend(random.sample(additional_refs, additional_count))
            
            items_to_process.append((client, commit, cluster_id, strategy_summary, commit_references))
    
    print(f"准备处理 {len(items_to_process)} 个commit的优化...")
    
    # 并行处理所有需要优化的commit
    if items_to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(
                executor.map(process_item, items_to_process),
                total=len(items_to_process),
                desc="优化函数"
            ))
        
        # 输出结果
        for commit_results in results:
            for result in commit_results:
                print(result)
    else:
        print("没有找到需要优化的commit")

if __name__ == "__main__":
    main()