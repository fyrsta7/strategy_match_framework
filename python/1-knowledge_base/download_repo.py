import os
import json
import subprocess
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# —— 全局配置 —— 
# 存放待克隆仓库列表的文件
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342_3.json")

# 并发线程数
MAX_WORKERS = 16

# 克隆方式：'ssh' 或 'http'
CLONE_URL_TYPE = 'ssh'

# —— 新增全局变量 —— 
# 指定所有仓库克隆后存放的根目录
# 默认放在 config.root_path/repository
DOWNLOAD_DIR = os.path.join(config.root_path, "repository")


def safe_remove_directory(path):
    """安全地删除目录，处理各种异常情况"""
    if not os.path.exists(path):
        return

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Could not remove directory {path} using shutil: {e}")
        try:
            if os.name != 'nt':
                subprocess.run(["chmod", "-R", "+w", path], check=False)
                subprocess.run(["rm", "-rf", path], check=False)
            else:
                subprocess.run(["rmdir", "/s", "/q", path], check=False, shell=True)
        except Exception as e2:
            print(f"Error: Failed to force delete {path}: {e2}")


def clone_repository(repo, download_root):
    """
    克隆单个代码库到 download_root。如果克隆失败，清理部分克隆的目录。
    """
    name = repo.get("name")
    if CLONE_URL_TYPE.lower() == 'ssh':
        url = repo.get("ssh_url")
    else:
        url = repo.get("http_url")

    if not name or not url:
        return f"Skipping invalid repository entry: {repo}"

    target_dir = os.path.join(download_root, name)

    # 若目标目录存在，检查是否为有效 Git 仓库
    if os.path.exists(target_dir):
        git_dir = os.path.join(target_dir, '.git')
        if os.path.exists(git_dir):
            try:
                res = subprocess.run(
                    ["git", "-C", target_dir, "rev-parse", "--is-inside-work-tree"],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                if res.stdout.decode().strip() == "true":
                    return f"Repository '{name}' already exists. Skipping."
            except subprocess.CalledProcessError:
                print(f"Found incomplete repo at '{target_dir}'. Re-cloning...")
                safe_remove_directory(target_dir)
        else:
            print(f"Directory '{target_dir}' exists but is not a Git repo. Removing...")
            safe_remove_directory(target_dir)

    # 确保目录已清理
    if os.path.exists(target_dir):
        return f"Could not remove existing directory '{target_dir}'. Skipping clone."

    try:
        subprocess.run(
            ["git", "clone", url, target_dir],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # 验证克隆结果
        git_dir = os.path.join(target_dir, '.git')
        if os.path.exists(git_dir):
            res = subprocess.run(
                ["git", "-C", target_dir, "rev-parse", "--is-inside-work-tree"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if res.stdout.decode().strip() == "true":
                return f"Successfully cloned '{name}' via {CLONE_URL_TYPE.upper()}: {url}"

        # 若验证失败，清理
        print(f"Clone of '{name}' incomplete. Cleaning up...")
        safe_remove_directory(target_dir)
        return f"Failed to properly clone '{name}'."

    except subprocess.CalledProcessError as e:
        if os.path.exists(target_dir):
            print(f"Clone failed. Cleaning up '{target_dir}'...")
            safe_remove_directory(target_dir)
        return f"Failed to clone '{name}'. Error: {e.stderr.decode().strip()}"

    except Exception as e:
        if os.path.exists(target_dir):
            print(f"Unexpected error. Cleaning up '{target_dir}'...")
            safe_remove_directory(target_dir)
        return f"Unexpected error while cloning '{name}': {e}"


def clone_repositories():
    # 使用全局 DOWNLOAD_DIR
    repository_dir = DOWNLOAD_DIR
    os.makedirs(repository_dir, exist_ok=True)

    print(f"Loading repositories from {REPO_LIST_FILE}...")
    print(f"Cloning into '{repository_dir}' using {CLONE_URL_TYPE.upper()} URLs.")

    try:
        with open(REPO_LIST_FILE, "r", encoding="utf-8") as f:
            repositories = json.load(f)
    except FileNotFoundError:
        print(f"Error: {REPO_LIST_FILE} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: {REPO_LIST_FILE} is not valid JSON.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(clone_repository, repo, repository_dir): repo.get("name")
            for repo in repositories
        }

        with tqdm(total=len(repositories), desc="Cloning repositories", unit="repo") as pbar:
            for future in as_completed(futures):
                name = futures[future] or "<unknown>"
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Error processing '{name}': {e}")
                    # 发生异常时清理
                    target = os.path.join(repository_dir, name)
                    if os.path.exists(target):
                        safe_remove_directory(target)
                pbar.update(1)

    print("All repositories processed.")


if __name__ == "__main__":
    clone_repositories()