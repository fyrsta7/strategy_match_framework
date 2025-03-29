import git
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 全局变量：
# 如果一个 commit 已经存在 modified_files_count 字段，是否仍强制重新计算
# False: 不重新计算
# True: 重新计算
RECALCULATE_IF_EXISTS = False

REPO_LIST_FILE = "repo_list_30345_3.json"
JSON_FILE = "all_commit.json"

def process_repository(repo_info):
    """
    处理单个代码仓库，更新 commit 信息并写回文件。
    """
    repo_name = repo_info["name"]
    print(repo_name)
    results = {"repo_name": repo_name, "status": "success", "error": None}
    try:
        # 仓库路径
        repo_path = os.path.join(config.root_path, "repository", repo_name)
        # print(repo_path)
        # 输出文件路径
        output_path = os.path.join(config.root_path, "knowledge_base", repo_name, JSON_FILE)
        # 如果输出文件不存在，则跳过该仓库
        if not os.path.exists(output_path):
            return {"repo_name": repo_name, "status": "skipped", "error": f"{output_path} not found"}
        # 读取 commit 信息
        with open(output_path, "r") as f:
            commit_data = json.load(f)
        # 打开本地仓库
        repo = git.Repo(repo_path)
        # 更新 commit 信息
        for commit in commit_data:
            # 如果 commit 已经存在 modified_files_count 字段，
            # 且全局变量设置为不重新计算，则跳过该 commit
            if "modified_files_count" in commit and not RECALCULATE_IF_EXISTS:
                continue

            commit_hash = commit["hash"]
            # print(commit["hash"])
            try:
                # 获取指定 commit 对象
                commit_obj = repo.commit(commit_hash)
                # 获取该 commit 修改的文件列表
                modified_files = commit_obj.stats.files
                # 获取所有被修改文件的相对路径
                modified_files_paths = list(modified_files.keys())
                # 更新 commit 信息
                commit["modified_files_count"] = len(modified_files)
                commit["modified_files"] = modified_files_paths
                # 构造 GitHub commit URL
                if "http_url" in repo_info:
                    http_url = repo_info["http_url"]
                    if http_url.endswith(".git"):
                        http_url = http_url[:-4]  # 去掉 .git 后缀
                    commit["github_commit_url"] = f"{http_url}/commit/{commit_hash}"
                else:
                    commit["github_commit_url"] = None
            except Exception as e:
                commit["modified_files_count"] = 0
                commit["modified_files"] = []
                commit["github_commit_url"] = None
        # 将更新后的数据写回文件
        with open(output_path, "w") as f:
            json.dump(commit_data, f, indent=4)
        return results
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        return results

def get_commit_file_count():
    """
    并行处理所有代码库，更新 commit 信息并写回文件。
    """
    # 配置并行线程数上限
    max_parallel_threads = 128  # 根据需求可以调整线程数上限
    # 配置文件路径
    repo_list_path = os.path.join(config.root_path, REPO_LIST_FILE)
    # 读取 repo_list.json
    with open(repo_list_path, "r") as f:
        repo_list = json.load(f)
    results = []
    # 使用 ProcessPoolExecutor 并行处理仓库，并限制线程数
    with ProcessPoolExecutor(max_workers=max_parallel_threads) as executor:
        # 提交所有任务
        futures = {executor.submit(process_repository, repo_info): repo_info for repo_info in repo_list}
        # 使用 tqdm 显示任务进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing repositories", unit="repo"):
            result = future.result()
            print(result)
            results.append(result)
            if result["status"] == "error":
                print(f"Error processing repository {result['repo_name']}: {result['error']}")
            elif result["status"] == "skipped":
                print(f"Skipped repository {result['repo_name']}: {result['error']}")
    print("所有仓库处理完成。")

if __name__ == "__main__":
    get_commit_file_count()
