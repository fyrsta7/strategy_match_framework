import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# 全局变量
FORCE_REPROCESS = False  # 设置为 True 时，强制重新处理；设置为 False 时，跳过已处理的结果
MAX_WORKERS = 4  # 并行数量，根据需求调整
TIMEOUT = 30  # 每个任务的最大运行时间（秒）

# 定义系统软件领域的提示词，并加入已知的五个系统软件代码库的示例
system_prompt = """
You are an expert in software classification. Your task is to determine whether a given code repository belongs to the system software domain. System software typically includes operating systems, device drivers, compilers, debuggers, file systems, virtualization tools, and other low-level software that interacts directly with hardware or provides foundational services for other software.

Here are some examples of repositories that belong to the system software domain:
1. **Ceph** (https://github.com/ceph/ceph): A scalable, distributed storage system designed to provide high-performance, reliable, and flexible storage solutions.
2. **Hyperscan** (https://github.com/intel/hyperscan): A high-performance regex matching library designed for simultaneous matching of large numbers of regular expressions across data streams.
3. **MySQL** (https://github.com/mysql/mysql-server): The official source code for MySQL, a widely-used open-source SQL database server.
4. **OpenSSL** (https://github.com/openssl/openssl): A robust, open-source toolkit for implementing the TLS, DTLS, and QUIC protocols, along with a comprehensive cryptographic library.
5. **RocksDB** (https://github.com/facebook/rocksdb): A high-performance persistent key-value store library optimized for flash and RAM storage.

You must answer strictly with "true" or "false":
- Answer "true" if the repository belongs to the system software domain.
- Answer "false" if the repository does not belong to the system software domain.

Do not answer with "unknown". You must provide a definitive answer ("true" or "false").
"""

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url=config.deepseek_base_url,
    api_key=config.deepseek_api_key,
)

def read_repo_list(json_file_path):
    """
    从 repo_list.json 中读取代码库列表。
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            repositories = json.load(f)
        return repositories
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} is not a valid JSON file.")
        return []

def write_repo_list(json_file_path, repositories):
    """
    将更新后的代码库列表写回 repo_list.json。
    """
    try:
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(repositories, f, indent=4)
        print(f"Updated repository list saved to {json_file_path}.")
    except Exception as e:
        print(f"Error writing to {json_file_path}: {e}")

def read_readme_file(repo_name, result_dir):
    """
    从 result/<repository name> 文件夹中读取 README.md 文件内容。
    """
    readme_path = os.path.join(result_dir, repo_name, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def query_llm(repo_name, repo_url, readme_content):
    """
    调用 OpenAI API 判断代码库是否属于系统软件领域。
    """
    # 构建用户提示
    user_prompt = f"""
    Here is the information for a code repository:
    - Name: {repo_name}
    - URL: {repo_url}
    - README Content: {readme_content if readme_content else "No README available."}

    Based on this information, does this repository belong to the system software domain?
    Please answer with "true" or "false".
    """

    # 调用 OpenAI API
    try:
        response = client.chat.completions.create(
            model=config.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=TIMEOUT,  # 设置超时时间
        )
        result = response.choices[0].message.content
        # 强制返回 "true" 或 "false"
        return "true" if result.lower() == "true" else "false"
    except Exception as e:
        print(f"Error querying LLM for {repo_name}: {e}")
        return "false"  # 如果出错，默认返回 "false"

def process_repository(repo, result_dir):
    """
    处理单个代码库。
    """
    repo_name = repo.get("name")
    repo_url = repo.get("http_url")

    if not repo_name or not repo_url:
        print(f"Skipping invalid repository entry: {repo}")
        return repo

    # 如果 FORCE_REPROCESS 为 False 且已经存在判断结果，跳过
    if not FORCE_REPROCESS and "is_system_software" in repo:
        print(f"Skipping {repo_name} (already processed).")
        return repo

    # 读取 README 文件
    readme_content = read_readme_file(repo_name, result_dir)

    # 调用 LLM 判断
    is_system_software = query_llm(repo_name, repo_url, readme_content)

    # 将结果添加到代码库信息中
    repo["is_system_software"] = is_system_software
    print(f"Processed {repo_name}: is_system_software = {is_system_software}")

    return repo

def check_known_repositories(repositories):
    """
    检查已知的五个系统软件代码库是否被正确分类为 "true"。
    """
    known_repos = {
        "ceph": "https://github.com/ceph/ceph.git",
        "hyperscan": "https://github.com/intel/hyperscan.git",
        "mysql-server": "https://github.com/mysql/mysql-server.git",
        "openssl": "https://github.com/openssl/openssl.git",
        "rocksdb": "https://github.com/facebook/rocksdb.git",
    }

    for repo in repositories:
        repo_name = repo.get("name")
        repo_url = repo.get("http_url")
        if repo_name.lower() in known_repos and repo_url == known_repos[repo_name.lower()]:
            if repo.get("is_system_software") != "true":
                print(f"Error: {repo_name} was not classified as 'true'.")
            else:
                print(f"Success: {repo_name} was correctly classified as 'true'.")

def main():
    # 定义路径
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # 获取 root_path
    json_file_path = os.path.join(root_path, "repo_list.json")  # repo_list.json 的路径
    result_dir = os.path.join(root_path, "result")  # result 文件夹的路径

    # 读取 repo_list.json
    repositories = read_repo_list(json_file_path)
    if not repositories:
        return

    # 使用 ThreadPoolExecutor 并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_repository, repo, result_dir)
            for repo in repositories
        ]

        # 使用 tqdm 显示进度
        updated_repositories = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing repositories"):
            try:
                result = future.result(timeout=TIMEOUT)  # 设置任务超时时间
                updated_repositories.append(result)
            except TimeoutError:
                print("A task timed out and was skipped.")

    # 将更新后的代码库列表写回 repo_list.json
    write_repo_list(json_file_path, updated_repositories)

    # 检查已知的五个系统软件代码库是否被正确分类
    check_known_repositories(updated_repositories)

if __name__ == "__main__":
    main()