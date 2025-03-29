import os
import json
import sys
import re
import concurrent.futures
from tqdm import tqdm
from colorama import Fore, Style, init
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 初始化colorama
init()

# 全局变量：设置是否在优化结果已存在时重新生成优化结果
# 设置为 True 时，即使文件存在，也会重新调用 LLM 生成优化结果
# 设置为 False 时，如果文件已存在，则跳过该提交的优化过程
REGENERATE_EXISTING = False

# 新增：每个函数要重复优化的次数
OPTIMIZATION_ATTEMPTS = 3

# 新增：当LLM回复不符合要求时，最大重试次数
MAX_RETRY_ATTEMPTS = 3

# prompt 中的例子数量
MAX_EXAMPLES = 4

# 使用的知识库版本
KNOWLEDGE_BASE_SEZE = 19998

# 定义需要处理的模型
MODELS = ['gpt-4o', 'deepseek-v3']

# 支持的模型列表
SUPPORTED_MODELS = ['deepseek-v3', 'deepseek-reasoner', 'gpt-4o', 'o1-mini', 'o3-mini']

# 放 commit 相关文件的路径
COMMIT_DIR = config.root_path + "benchmark/{repo}/modified_file/{commit}"

# 将优化结果放在 COMMIT_DIR/OUTPUT_DIR/<model name>/ 这个子文件夹中
OUTPUT_DIR = "bm25"

# 加载全局的相似性数据
similarity_path = os.path.join(config.root_path, f"commit_similarity_bm25_{KNOWLEDGE_BASE_SEZE}_top20.json")

# 指定读取需要被优化的 commit 列表文件（文件内容为 commit 对象数组，包含 "hash" 等信息）
COMMIT_LIST_FILE = os.path.join(config.root_path, 'benchmark', 'rocksdb', 'filtered_test_result.json')

def extract_code_blocks(text):
    """
    从文本中提取所有代码块
    返回: 列表，包含所有找到的代码块内容（不含围栏）
    """
    # 匹配格式为 ```language code ``` 的代码块，忽略语言标识
    pattern = r"```(?:[a-zA-Z0-9+]*\n)?(.*?)```"
    # re.DOTALL 标志使 . 能匹配包括换行符在内的所有字符
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def get_client_and_model_info(model):
    """
    根据传入的模型名称返回 (client, model_name)。
    注意：不同模型需要不同的调用参数和变量。
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Error: Unsupported model '{model}'. Supported: {SUPPORTED_MODELS}")
    
    if model == 'deepseek-v3':
        model_name = "volc/deepseek-v3-241226"
    elif model == 'deepseek-reasoner':
        model_name = "volc/deepseek-r1-250120"
    else:
        # 对于 'gpt-4o'、'o1-mini' 或 'o3-mini'
        model_name = "closeai/" + model
    
    try:
        client = OpenAI(
            base_url=config.xmcp_base_url,
            api_key=config.xmcp_api_key,
        )
    except Exception as e:
        raise RuntimeError(f"Client initialization failed for model {model}: {e}")
    
    return client, model_name

def load_example_data(commit_info):
    """Load example data (before_func and diff)"""
    result_dir = os.path.join(
        config.root_path,
        'knowledge_base',
        commit_info["repository_name"],
        'modified_file',
        commit_info["commit_hash"]
    )
    
    data = {}
    try:
        with open(os.path.join(result_dir, 'before_func.txt'), 'r', encoding='utf-8') as f:
            data['before'] = f.read().strip()
        with open(os.path.join(result_dir, 'diff.txt'), 'r', encoding='utf-8') as f:
            data['diff'] = f.read().strip()
        return data
    except Exception as e:
        print(f"Failed to load example {commit_info['commit_hash']}: {str(e)}")
        return None

def build_examples_section(commit_hash, repo_name, similarity_data):
    """Build examples section (ascending similarity order)"""
    query_entry = next(
        (item for item in similarity_data 
         if item["query_commit"]["commit_hash"] == commit_hash
         and item["query_commit"]["repository_name"] == repo_name),
        None
    )
    
    if not query_entry:
        return ""
    
    # 不再根据相似度过滤，直接采用所有示例
    valid_examples = query_entry["similar_commits"]
    
    sorted_examples = sorted(
        valid_examples,
        key=lambda x: x["similarity_score"],
        reverse=True
    )[:MAX_EXAMPLES][::-1]  # 保证按相似性升序排列
    
    examples = []
    for ex in sorted_examples:
        if (ex_data := load_example_data(ex)):
            examples.append({
                "similarity": ex["similarity_score"],
                "before": ex_data['before'],
                "diff": ex_data['diff']
            })
    
    example_text = ""
    for idx, ex in enumerate(examples):
        example_text += (
            f"==== Example {idx+1} (Similarity: {ex['similarity']:.1f}) ====\n"
            "Original Function:\n"
            f"{ex['before']}\n\n"
            "Optimization Changes:\n"
            f"```diff\n{ex['diff']}\n```\n"
            "------------------------\n\n"
        )
    
    return example_text.strip()

def generate_prompt(target_function, examples_section):
    """Generate optimized English prompt"""
    return f"""Optimize the following C/C++ function based on the reference examples. Requirements:
1. Maintain EXACT functionality
2. Place your complete optimized function in a single code block using triple backticks (```)
3. You may include explanations of your optimization approach, but ensure there is only ONE code block in your reply
Reference optimization examples (ordered by ascending similarity):
{examples_section if examples_section else "No relevant examples"}
Function to optimize:
```cpp
{target_function}
```
Return your optimized function with explanations if needed, but make sure to include exactly ONE code block with the complete optimized code."""

def optimize_function_single_attempt(client, repo_name, commit_hash, similarity_data, current_model, model_name, attempt_number):
    """
    单次调用LLM进行代码优化
    """
    # 构建存储路径 - 修改为使用模型名作为子目录
    commit_dir = COMMIT_DIR.format(repo=repo_name, commit=commit_hash)
    
    # 创建输出目录，以模型名称作为子文件夹
    output_dir = os.path.join(commit_dir, OUTPUT_DIR, current_model)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径，使用数字作为文件名
    output_path = os.path.join(output_dir, f"{attempt_number}.txt")
    
    # 如果结果文件已存在且不允许重新生成，则跳过该提交
    if os.path.exists(output_path) and not REGENERATE_EXISTING:
        return {
            "status": "skipped",
            "message": f"Skipping existing commit: {commit_hash} (model: {current_model}, attempt: {attempt_number})",
            "output_path": output_path,
            "commit_hash": commit_hash
        }
    
    # 读取待优化的原始函数
    before_func_path = os.path.join(commit_dir, 'before_func.txt')
    if not os.path.exists(before_func_path):
        return {
            "status": "error",
            "message": f"Missing source file: {before_func_path}",
            "output_path": None,
            "commit_hash": commit_hash
        }
    
    try:
        with open(before_func_path, 'r', encoding='utf-8') as f:
            target_function = f.read().strip()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read target function for commit {commit_hash}: {str(e)}",
            "output_path": None,
            "commit_hash": commit_hash
        }
    
    # 构建 prompt：包括示例部分与目标函数
    examples_section = build_examples_section(commit_hash, repo_name, similarity_data)
    full_prompt = generate_prompt(target_function, examples_section)
    
    messages = [{"role": "user", "content": full_prompt}]
    
    # 尝试调用LLM，并在必要时重试
    for retry_count in range(MAX_RETRY_ATTEMPTS):
        try:
            if current_model in ["o1-mini", "o3-mini"]:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    # temperature=0,
                    # max_tokens=8192
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=8192
                )
            
            response_content = response.choices[0].message.content.strip()
            
            # 提取代码块
            code_blocks = extract_code_blocks(response_content)
            
            # 检查代码块数量
            if len(code_blocks) == 0:
                if retry_count < MAX_RETRY_ATTEMPTS - 1:
                    messages.append({"role": "assistant", "content": response_content})
                    messages.append({
                        "role": "user", 
                        "content": "You didn't include a code block with your optimized function. Please provide your complete optimized function in a single markdown code block using triple backticks (```)."
                    })
                    continue
                else:
                    return {
                        "status": "error",
                        "message": f"Error: Commit {commit_hash}, attempt {attempt_number} failed to generate code block after {MAX_RETRY_ATTEMPTS} retries.",
                        "output_path": None,
                        "commit_hash": commit_hash
                    }
            
            elif len(code_blocks) > 1:
                if retry_count < MAX_RETRY_ATTEMPTS - 1:
                    messages.append({"role": "assistant", "content": response_content})
                    messages.append({
                        "role": "user", 
                        "content": "You included multiple code blocks. Please provide only ONE code block with your complete optimized function."
                    })
                    continue
                else:
                    return {
                        "status": "error",
                        "message": f"Error: Commit {commit_hash}, attempt {attempt_number} generated multiple code blocks after {MAX_RETRY_ATTEMPTS} retries.",
                        "output_path": None,
                        "commit_hash": commit_hash
                    }
            
            # 获取提取的代码
            optimized_code = code_blocks[0]
            
            # 保存优化结果至文件
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_code)
                status_text = "Regenerated" if os.path.exists(output_path) else "Created"
                return {
                    "status": "success",
                    "message": f"{status_text} optimization for commit {commit_hash} (model: {current_model}, attempt: {attempt_number})",
                    "output_path": output_path,
                    "commit_hash": commit_hash
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to write output file for commit {commit_hash}: {str(e)}",
                    "output_path": None,
                    "commit_hash": commit_hash
                }
            
        except Exception as e:
            if retry_count < MAX_RETRY_ATTEMPTS - 1:
                continue
            return {
                "status": "error",
                "message": f"API request failed for commit {commit_hash} (model: {current_model}, attempt: {attempt_number}, retry: {retry_count+1}): {str(e)}",
                "output_path": None,
                "commit_hash": commit_hash
            }
    
    return {
        "status": "error",
        "message": f"Error: All optimization attempts failed for commit {commit_hash}, attempt {attempt_number}",
        "output_path": None,
        "commit_hash": commit_hash
    }

def optimize_function(client, repo_name, commit_hash, similarity_data, current_model, model_name):
    """
    对函数进行多次优化，每次产生一个单独的结果文件
    """
    results = []
    
    for attempt in range(1, OPTIMIZATION_ATTEMPTS + 1):
        result = optimize_function_single_attempt(
            client, repo_name, commit_hash, similarity_data, current_model, model_name, attempt
        )
        results.append(result)
    
    return results

def process_commit(args):
    """用于并行处理的包装函数"""
    client, repo_name, commit_obj, similarity_data, current_model, model_name = args
    commit_hash = commit_obj.get("hash")
    if not commit_hash:
        return [{
            "status": "error", 
            "message": "Warning: commit object missing 'hash' field, skipping.",
            "output_path": None,
            "commit_hash": None
        }]
    
    # 在 commit_similarity.json 中查找对应的记录
    query_entry = next(
        (item for item in similarity_data
         if item["query_commit"]["commit_hash"] == commit_hash \
            and item["query_commit"]["repository_name"] == repo_name),
        None
    )
    if not query_entry:
        return [{
            "status": "error",
            "message": f"Commit {commit_hash} not found in similarity data, skipping.",
            "output_path": None,
            "commit_hash": commit_hash
        }]
    
    return optimize_function(client, repo_name, commit_hash, similarity_data, current_model, model_name)

def print_result_with_color(result):
    """格式化输出结果，带有颜色标记"""
    status = result["status"]
    message = result["message"]
    output_path = result["output_path"]
    commit_hash = result["commit_hash"]
    
    if status == "success":
        status_color = Fore.GREEN
        path_info = f"\n  📁 Result saved to: {output_path}"
    elif status == "skipped":
        status_color = Fore.YELLOW
        path_info = f"\n  📁 Existing file path: {output_path}"
    else:  # error
        status_color = Fore.RED
        path_info = ""
    
    print(f"{status_color}[{status.upper()}]{Style.RESET_ALL} {message}{path_info}")

def summarize_results(results_by_commit, model):
    """生成并打印结果摘要"""
    success_count = 0
    skipped_count = 0
    error_count = 0
    total_attempts = 0
    
    for commit_hash, attempts in results_by_commit.items():
        for result in attempts:
            total_attempts += 1
            if result["status"] == "success":
                success_count += 1
            elif result["status"] == "skipped":
                skipped_count += 1
            else:  # error
                error_count += 1
    
    success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0
    
    print("\n" + "=" * 70)
    print(f"SUMMARY FOR MODEL: {model}")
    print("=" * 70)
    print(f"Total commits processed: {len(results_by_commit)}")
    print(f"Total optimization attempts: {total_attempts}")
    print(f"{Fore.GREEN}Successful optimizations: {success_count} ({success_rate:.1f}%){Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Skipped optimizations: {skipped_count}{Style.RESET_ALL}")
    print(f"{Fore.RED}Failed optimizations: {error_count}{Style.RESET_ALL}")
    print("=" * 70 + "\n")

def main():
    repo_name = "rocksdb"
    
    try:
        with open(similarity_path, 'r', encoding='utf-8') as f:
            similarity_data = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Failed to load similarity data: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)
    
    # 加载需要优化的 commit 列表，文件中每个元素为一个 commit 对象，其中必须包含 "hash" 字段
    try:
        with open(COMMIT_LIST_FILE, 'r', encoding='utf-8') as f:
            commit_list = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Failed to load commit list from {COMMIT_LIST_FILE}: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"Loaded {len(commit_list)} commits to process")
    print(f"Will optimize each function {OPTIMIZATION_ATTEMPTS} times with up to {MAX_RETRY_ATTEMPTS} retries per attempt")
    print(f"Results will be saved to <commit_hash>/{OUTPUT_DIR}/<model_name> directory for each commit")
    
    # 针对每个模型进行处理
    for current_model in MODELS:
        print(f"\n{Fore.CYAN}=================== Processing with model {current_model} ==================={Style.RESET_ALL}")
        
        try:
            client, model_name = get_client_and_model_info(current_model)
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize client for model {current_model}: {e}{Style.RESET_ALL}")
            continue
        
        # 准备并行处理的参数
        args_list = [(client, repo_name, commit_obj, similarity_data, current_model, model_name) 
                     for commit_obj in commit_list]
        
        # 使用进度条和并行执行
        results_all = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results_all = list(tqdm(
                executor.map(process_commit, args_list),
                total=len(args_list),
                desc=f"Optimization Progress ({current_model})"
            ))
        
        # 组织结果按提交分组
        results_by_commit = {}
        for commit_results in results_all:
            for result in commit_results:
                commit_hash = result["commit_hash"]
                if commit_hash:
                    if commit_hash not in results_by_commit:
                        results_by_commit[commit_hash] = []
                    results_by_commit[commit_hash].append(result)
        
        # 输出详细结果
        print(f"\n{Fore.CYAN}Detailed Results for {current_model}:{Style.RESET_ALL}")
        for commit_hash, results in results_by_commit.items():
            print(f"\n{Fore.BLUE}Commit: {commit_hash}:{Style.RESET_ALL}")
            for result in results:
                print_result_with_color(result)
        
        # 打印摘要
        summarize_results(results_by_commit, current_model)

if __name__ == "__main__":
    main()
