import os
import json
import sys
import re
import concurrent.futures
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 全局变量：设置是否在文件已存在时重新生成优化结果
# 设置为 True 时，即使文件存在，也会重新调用 LLM 生成优化结果
# 设置为 False 时，如果文件已存在，则跳过该提交的优化过程
REGENERATE_IF_EXISTS = False

# 每个函数需要重复优化的次数
OPTIMIZATION_ATTEMPTS = 3

# 新增：当LLM回复不符合要求时，最大重试次数
MAX_RETRY_ATTEMPTS = 3

# 定义所有需要处理的模型（请确保列表中的模型均在 SUPPORTED_MODELS 内）
MODELS = ['deepseek-v3', 'gpt-4o']

# 支持的模型列表
SUPPORTED_MODELS = ['deepseek-v3', 'deepseek-reasoner', 'gpt-4o', 'o1-mini']

# 指定从哪个文件中读取需要被优化的 commit 列表（文件中每个 commit 对象必须包含 "hash" 字段）
COMMIT_LIST_FILE = os.path.join(
    config.root_path,
    'benchmark',
    'rocksdb',
    'filtered_test_result.json'
)

def get_client_and_model_info(model):
    """
    根据传入的模型名称返回 (client, model_name)。
    注意：不同模型需要不同的调用参数和变量。
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型 '{model}'。支持的模型有：{SUPPORTED_MODELS}")
    
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
        raise RuntimeError(f"无法初始化 OpenAI 客户端。{e}")
    
    return client, model_name

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

def optimize_function_single_attempt(client, repo_name, commit_hash, current_model, model_name, attempt_number):
    """
    单次调用对应 LLM 客户端对函数进行优化
    """
    # 构建提交对应的文件目录路径
    benchmark_dir = os.path.join(
        config.root_path,
        'benchmark',
        repo_name,
        'modified_file',
        commit_hash
    )
    
    before_func_path = os.path.join(benchmark_dir, 'before_func.txt')
    
    # 新的存储路径结构：<commit hash>/baseline/<model>/
    model_dir = os.path.join(benchmark_dir, "baseline", current_model)
    os.makedirs(model_dir, exist_ok=True)
    result_file_path = os.path.join(model_dir, f"{attempt_number}.txt")
    
    # 跳过已存在的结果（除非允许重生成）
    if os.path.exists(result_file_path):
        if not REGENERATE_IF_EXISTS:
            return f"跳过提交 {commit_hash} 的第 {attempt_number} 次优化，因为 {result_file_path} 已存在。"
    
    # 检查 before_func.txt 是否存在
    if not os.path.exists(before_func_path):
        return f"错误：文件 {before_func_path} 不存在。"
    
    try:
        with open(before_func_path, 'r', encoding='utf-8') as f:
            function_content = f.read()
    except Exception as e:
        return f"错误：无法读取文件 {before_func_path}。{e}"
    
    # 修改后的提示，允许解释但要求包含单个代码块
    prompt = (
        "You are a helpful assistant. "
        "Please optimize the following C or C++ function for better performance. "
        "Ensure that the semantics of the function remain unchanged and the performance is improved. "
        "Only modify the code if necessary to change its logic or execution behavior. "
        "Avoid making any formatting changes that do not affect the code's functional behavior, such as altering indentation. "
        "Do not remove or modify any comments in the original code. "
        "Do not consider code readability or maintainability. "
        "You can explain your optimization approach, but make sure to include exactly ONE code block "
        "containing the complete optimized function. "
        "Do NOT include multiple code blocks or code snippets in your explanation. "
        "Place the full optimized function in a single markdown code block using triple backticks (```). "
        "Here is the function to optimize:\n\n"
        + function_content
    )
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
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
                    return f"错误：提交 {commit_hash} 的第 {attempt_number} 次优化经过 {MAX_RETRY_ATTEMPTS} 次尝试后仍未生成代码块"
            
            elif len(code_blocks) > 1:
                if retry_count < MAX_RETRY_ATTEMPTS - 1:
                    messages.append({"role": "assistant", "content": response_content})
                    messages.append({
                        "role": "user", 
                        "content": "You included multiple code blocks. Please provide only ONE code block with your complete optimized function."
                    })
                    continue
                else:
                    return f"错误：提交 {commit_hash} 的第 {attempt_number} 次优化经过 {MAX_RETRY_ATTEMPTS} 次尝试后仍生成了多个代码块"
            
            # 如果只有一个代码块，则保存结果
            optimized_function = code_blocks[0]
            
            try:
                with open(result_file_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_function)
                return f"提交 {commit_hash}: 第 {attempt_number} 次优化后的函数已保存到 {result_file_path}"
            except Exception as e:
                return f"错误：无法写入文件 {result_file_path}。{e}"
        
        except Exception as e:
            if retry_count < MAX_RETRY_ATTEMPTS - 1:
                continue
            return f"请求 OpenAI 时出错（Commit: {commit_hash}，模型: {current_model}，尝试: {attempt_number}，重试: {retry_count + 1}）：{e}"
    
    return f"错误：提交 {commit_hash} 的第 {attempt_number} 次优化全部失败"

def optimize_function(client, repo_name, commit_hash, current_model, model_name):
    """
    对函数进行多次优化，每次产生一个单独的结果文件
    """
    results = []
    
    for attempt in range(1, OPTIMIZATION_ATTEMPTS + 1):
        result = optimize_function_single_attempt(
            client, repo_name, commit_hash, current_model, model_name, attempt
        )
        results.append(result)
    
    return results

def process_commit(args):
    """用于并行处理的包装函数"""
    client, repo_name, commit, current_model, model_name = args
    commit_hash = commit.get("hash")
    if not commit_hash:
        return ["警告：提交记录缺少 'hash' 字段，跳过该记录。"]
    
    return optimize_function(client, repo_name, commit_hash, current_model, model_name)

def main():
    repo_name = "rocksdb"  # 如有需要，可修改为其他代码库名称
    
    # 从 COMMIT_LIST_FILE 中加载需要被优化的 commit 列表
    try:
        with open(COMMIT_LIST_FILE, 'r', encoding='utf-8') as f:
            commit_list = json.load(f)
    except Exception as e:
        print(f"错误：无法读取 commit 列表文件 {COMMIT_LIST_FILE}。{e}")
        sys.exit(1)
    
    # 针对列表中的每个 LLM 进行处理
    for current_model in MODELS:
        print(f"\n==================== 正在使用模型 {current_model} 进行优化处理 ====================")
        print(f"每个函数将进行 {OPTIMIZATION_ATTEMPTS} 次优化尝试，每次尝试最多重试 {MAX_RETRY_ATTEMPTS} 次")
        
        # 获取对应的客户端与调用时的 model_name
        try:
            client, model_name = get_client_and_model_info(current_model)
        except Exception as e:
            print(f"模型 {current_model} 初始化失败：{e}")
            continue
        
        # 准备并行处理的参数
        args_list = [(client, repo_name, commit, current_model, model_name) 
                     for commit in commit_list]
        
        # 使用进度条和并行执行
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(process_commit, args_list),
                total=len(args_list),
                desc=f"优化进度 ({current_model})"
            ))
        
        # 输出结果
        for commit_idx, commit_results in enumerate(results):
            for result in commit_results:
                print(result)

if __name__ == "__main__":
    main()