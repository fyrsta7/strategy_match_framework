import os
import json
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30345_3.json")  # repository_list.json 的路径

def clone_repository(repo, repository_dir):
    """
    克隆单个代码库。
    """
    name = repo.get("name")
    url = repo.get("ssh_url")

    if not name or not url:
        return f"Skipping invalid repository entry: {repo}"

    target_dir = os.path.join(repository_dir, name)

    # 如果目标文件夹已存在，跳过克隆
    if os.path.exists(target_dir):
        return f"Repository '{name}' already exists at '{target_dir}'. Skipping clone."

    try:
        # 执行 git clone 命令
        subprocess.run(["git", "clone", url, target_dir], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"Successfully cloned '{name}'."
    except subprocess.CalledProcessError as e:
        return f"Failed to clone '{name}'. Error: {e.stderr.decode().strip()}"
    except Exception as e:
        return f"An unexpected error occurred while cloning '{name}'. Error: {str(e)}"

def clone_repositories():
    # 定义路径
    repository_dir = os.path.join(config.root_path, "repository")  # repository 文件夹的路径

    # 确保 repository 文件夹存在
    os.makedirs(repository_dir, exist_ok=True)

    # 读取 repository_list.json 文件
    print(f"Loading repositories from {REPO_LIST_FILE}...")
    try:
        with open(REPO_LIST_FILE, "r", encoding="utf-8") as f:
            repositories = json.load(f)
    except FileNotFoundError:
        print(f"Error: {REPO_LIST_FILE} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: {REPO_LIST_FILE} is not a valid JSON file.")
        return

    # 使用线程池并行克隆代码库
    with ThreadPoolExecutor(max_workers=64) as executor:
        # 提交任务
        futures = {
            executor.submit(clone_repository, repo, repository_dir): repo.get("name")
            for repo in repositories
        }

        # 使用 tqdm 显示进度
        with tqdm(total=len(repositories), desc="Cloning repositories", unit="repo") as pbar:
            for future in as_completed(futures):
                repo_name = futures[future]
                try:
                    result = future.result()
                    print(result)  # 打印每个任务的结果
                except Exception as e:
                    print(f"Error processing '{repo_name}': {str(e)}")
                pbar.update(1)  # 更新进度条

    print("All repositories processed.")

if __name__ == "__main__":
    clone_repositories()