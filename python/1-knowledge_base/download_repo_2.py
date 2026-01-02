import os
import json
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")

# 定义两个存储路径
# 现有的存储代码库的路径
REPOSITORY_DIR_1 = "/ssd/zyw/llm_on_code/llm_on_code_optimization/repository/"
# 存储新下载的代码库的路径
REPOSITORY_DIR_2 = "/raid/zyw/llm_on_code/llm_on_code_optimization/repository/"

def clone_repository(repo):
    """
    克隆单个代码库，优先检查路径1，若不存在则克隆到路径2。
    """
    name = repo.get("name")
    url = repo.get("ssh_url")
    if not name or not url:
        return f"Skipping invalid repository entry: {repo}"
    
    # 检查路径1中是否已存在
    target_dir_1 = os.path.join(REPOSITORY_DIR_1, name)
    if os.path.exists(target_dir_1):
        return f"Repository '{name}' already exists at '{target_dir_1}'. Skipping clone."
    
    # 确定克隆的目标路径为路径2
    target_dir_2 = os.path.join(REPOSITORY_DIR_2, name)
    if os.path.exists(target_dir_2):
        return f"Repository '{name}' already exists at '{target_dir_2}'. Skipping clone."
    
    try:
        # 执行 git clone 命令，克隆到路径2
        subprocess.run(["git", "clone", url, target_dir_2], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"Successfully cloned '{name}' to {target_dir_2}."
    except subprocess.CalledProcessError as e:
        return f"Failed to clone '{name}'. Error: {e.stderr.decode().strip()}"
    except Exception as e:
        return f"An unexpected error occurred while cloning '{name}'. Error: {str(e)}"

def clone_repositories():
    # 确保两个存储文件夹都存在
    os.makedirs(REPOSITORY_DIR_1, exist_ok=True)
    os.makedirs(REPOSITORY_DIR_2, exist_ok=True)
    
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
    
    print(f"Repository paths:")
    print(f"  Primary path: {REPOSITORY_DIR_1}")
    print(f"  Secondary path: {REPOSITORY_DIR_2}")
    
    # 使用线程池并行克隆代码库
    with ThreadPoolExecutor(max_workers=24) as executor:
        # 提交任务
        futures = {
            executor.submit(clone_repository, repo): repo.get("name")
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