import os
import json
import sys
import concurrent.futures
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 全局变量：如果当前 commit 的 summary.txt 已存在是否重新生成（True：重新生成；False：跳过）
REGENERATE_SUMMARY = False

# 输入文件：related_commit.json 数据，其中每个条目包含 top_similar_commits 字段，
# 每个 commit 对象中至少包含 commit hash 以及 repository_name（若缺失则默认使用 "rocksdb"）
RELATED_COMMIT_FILE = os.path.join(config.root_path, 'benchmark', 'rocksdb', 'related_commit_langchain+cosine.json')

# 全局变量：当前使用的 LLM 模型，可修改为 'deepseek-v3'、'deepseek-reasoner'、'gpt-4o'、'o1-mini' 等
CURRENT_MODEL = 'gpt-4o'
# 支持的模型列表
SUPPORTED_MODELS = ['deepseek-v3', 'deepseek-reasoner', 'gpt-4o', 'o1-mini', 'o3-mini']

# 用于缓存已加载的 one_func.json 数据，避免重复读取同一代码库下的文件
one_func_cache = {}

def get_client_and_model_info(model):
    """
    根据传入的模型名称初始化并返回 (client, model_name) 对象。
    不同模型可能需要不同的 API 参数。
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型 '{model}'。支持的模型有：{SUPPORTED_MODELS}")
    
    if model == 'deepseek-v3':
        base_url = config.deepseek_base_url
        api_key = config.deepseek_api_key
        model_name = 'deepseek-chat'
    elif model == 'deepseek-reasoner':
        base_url = config.deepseek_base_url
        api_key = config.deepseek_api_key
        model_name = 'deepseek-reasoner'
    else:  # 针对 'gpt-4o' 或 'o1-mini'
        base_url = config.closeai_base_url
        api_key = config.closeai_api_key
        model_name = model

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    except Exception as e:
        raise RuntimeError(f"无法初始化 OpenAI 客户端：{e}")
    return client, model_name

def load_json_file(file_path, description=""):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"错误：无法加载 {description} 文件 {file_path}。{e}")
        return None

def load_commit_files(repo_name, commit_hash):
    """
    根据 repository_name 与 commit_hash，从路径 root_path/result/<repository_name>/modified_file/<commit_hash>/ 
    下加载 before_func.txt 与 diff.txt 的内容；返回字典：{"before": 内容, "diff": 内容}，失败时返回 None。
    """
    base_dir = os.path.join(config.root_path, 'result', repo_name, 'modified_file', commit_hash)
    before_func_path = os.path.join(base_dir, 'before_func.txt')
    diff_path = os.path.join(base_dir, 'diff.txt')
    
    try:
        with open(before_func_path, 'r', encoding='utf-8') as f:
            before_content = f.read().strip()
    except Exception as e:
        print(f"警告：无法读取文件 {before_func_path}。{e}")
        return None
    try:
        with open(diff_path, 'r', encoding='utf-8') as f:
            diff_content = f.read().strip()
    except Exception as e:
        print(f"警告：无法读取文件 {diff_path}。{e}")
        return None
    
    return {"before": before_content, "diff": diff_content}

def retrieve_commit_message(repo_name, commit_hash):
    """
    根据 repository_name 与 commit_hash，从路径 root_path/result/<repository_name>/one_func.json 中查找对应 commit，
    并返回 commit 的 message；若找不到则返回空字符串。
    利用缓存 one_func_cache 避免重复加载同一文件。
    """
    global one_func_cache
    one_func_path = os.path.join(config.root_path, 'result', repo_name, 'one_func.json')
    if repo_name not in one_func_cache:
        try:
            with open(one_func_path, 'r', encoding='utf-8') as f:
                one_func_cache[repo_name] = json.load(f)
        except Exception as e:
            print(f"错误：无法加载 {one_func_path}。{e}")
            one_func_cache[repo_name] = []
    for item in one_func_cache[repo_name]:
        if item.get("hash") == commit_hash:
            return item.get("message", "")
    return ""

def build_prompt(commit_msg, before_func, diff_content):
    """
    构造调用LLM的 prompt，内容包含 commit message、修改前函数代码与 git diff 信息，
    并附上详细的英文说明，要求LLM根据分类系统给出优化原因、实现方式及适用场景的总结。
    提示（prompt）全部使用英文。
    """
    prompt = f"""
Below are the details of a commit that performed a performance optimization.

Commit Message:
{commit_msg}

Before Function:
{before_func}

Diff:
{diff_content}

Please analyze the commit based on the following classification system and summarize its main optimization strategy.

### Classification System

1. Reason for performance issue:
   - Which category does the main problem belong to?
     - Structural
     - Misunderstanding/Misuse
     - Missing Operation
     - Redundancy or Memoization Opportunity
     - Others
   - If Structural is chosen, specify further:
     - Condition, Loop, Data Structure, Synchronization, or Query.
   - If Misunderstanding/Misuse is chosen, specify further:
     - API Call, Inter-procedural issues, Unnecessary Computation, or Configuration.
   - For Missing Operation, Redundancy, or Others, no further classification is needed.

2. Impact on performance:
   - Which aspect is mainly improved?
     - Execution Time, Memory Utilization, Energy Consumption, or No Significant Improvement.

3. Scope of application:
   - What kind of project is involved?
     - Server, Web, GUI, Database, Communication, Multiple Processes, System Software, or Others.
   - Is the optimization general or language-specific?
     - If language-specific, specify the programming language (e.g., C++, Java, Python, etc.).

Finally, provide a concise summary explaining:
- The specific performance issue addressed,
- How the optimization is implemented,
- And potential application scenarios.

Return your answer in plain text with clear sections for classifications and the final summary.
""".strip()
    return prompt

def call_llm(client, model_name, prompt):
    messages = [{"role": "user", "content": prompt}]
    try:
        if model_name == "o1-mini" or model_name == "o3-mini":
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=8192
            )
        result = response.choices[0].message.content.strip()
        # 去除返回结果中可能出现的 Markdown 代码块标记
        lines = result.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith('```') and lines[-1].strip().startswith('```'):
            result = "\n".join(lines[1:-1]).strip()
        return result
    except Exception as e:
        print(f"LLM 请求失败：{e}")
        return ""

def process_commit(key, commit_detail, client, model_name):
    """
    单个 commit 的处理逻辑，读取必要文件、构造 prompt，调用 LLM 并将结果写入 summary.txt，
    返回处理是否成功。
    """
    # key 格式为 repoName_commitHash
    parts = key.split("_", 1)
    repo_name = parts[0]
    commit_hash = parts[1]
    
    commit_dir = os.path.join(config.root_path, 'result', repo_name, 'modified_file', commit_hash)
    summary_file = os.path.join(commit_dir, 'summary.txt')
    if os.path.exists(summary_file) and not REGENERATE_SUMMARY:
        print(f"commit {repo_name}_{commit_hash} 的 summary.txt 已存在，跳过...")
        return False

    file_data = load_commit_files(repo_name, commit_hash)
    if file_data is None:
        print(f"跳过 commit {repo_name}_{commit_hash}，缺少必要文件。")
        return False

    commit_msg = retrieve_commit_message(repo_name, commit_hash)
    if not commit_msg:
        print(f"警告：commit {repo_name}_{commit_hash} 在 one_func.json 中未找到 commit message。")
    
    prompt = build_prompt(commit_msg, file_data["before"], file_data["diff"])
    summary = call_llm(client, model_name, prompt)
    if summary:
        os.makedirs(commit_dir, exist_ok=True)
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"commit {repo_name}_{commit_hash} 的优化总结已写入 {summary_file}")
            return True
        except Exception as e:
            print(f"错误：写入 {summary_file} 失败。{e}")
            return False
    else:
        print(f"commit {repo_name}_{commit_hash} 调用 LLM 未获得有效结果。")
        return False

def main():
    related_data = load_json_file(RELATED_COMMIT_FILE, "related commit")
    if related_data is None:
        sys.exit(1)
    
    # 构建 commit_detail_map，键为 "repositoryName_commitHash"，值为 commit 对象
    commit_detail_map = {}
    for entry in related_data:
        top_commits = entry.get("top_similar_commits", [])
        for commit in top_commits:
            commit_hash = commit.get("commit_hash") or commit.get("hash")
            repo_name = commit.get("repository_name", "rocksdb")
            if commit_hash:
                key = f"{repo_name}_{commit_hash}"
                if key not in commit_detail_map:
                    commit_detail_map[key] = commit
                    
    if not commit_detail_map:
        print("没有从相关数据中提取到 commit 信息。")
        sys.exit(0)
    
    total = len(commit_detail_map)
    print(f"共检测到 {total} 个 commit，将进行性能优化总结（根据 REGENERATE_SUMMARY = {REGENERATE_SUMMARY} 规则）")
    
    try:
        client, model_name = get_client_and_model_info(CURRENT_MODEL)
    except Exception as e:
        print(f"初始化 LLM 失败：{e}")
        sys.exit(1)
    
    processed = 0
    # 设置并行的最大线程数，可根据实际情况调整
    max_workers = 8
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有 commit 处理任务
        for key, commit_detail in commit_detail_map.items():
            futures.append(executor.submit(process_commit, key, commit_detail, client, model_name))
        
        # 使用 tqdm 显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing commits"):
            try:
                result = future.result()
                if result:
                    processed += 1
            except Exception as e:
                print(f"处理过程中出现异常：{e}")
    
    print(f"\n共处理 {processed} 个 commit 的优化总结。")

if __name__ == "__main__":
    main()
