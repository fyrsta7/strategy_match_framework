#!/usr/bin/env python3
import os
import sys
import concurrent.futures
from tqdm import tqdm

# 导入配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def remove_lock_for_repo(repo):
    """
    对单个仓库删除 .git/index.lock 文件（如果存在），并返回处理结果信息。
    """
    repo_path = os.path.join(repository_root, repo)
    if os.path.isdir(repo_path):
        lock_file = os.path.join(repo_path, ".git", "index.lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                return f"已删除 {lock_file}"
            except Exception as e:
                return f"删除 {lock_file} 时出错: {e}"
        else:
            return f"{lock_file} 不存在，跳过。"
    return f"{repo_path} 非目录，跳过。"

def remove_index_lock(repository_root):
    """
    遍历 repository_root 目录下的所有子目录，
    并并行删除每个仓库中的 .git/index.lock 文件（如果存在）。
    使用 tqdm 显示删除进度。
    """
    if not os.path.exists(repository_root):
        print(f"目录 {repository_root} 不存在。")
        return

    repos = os.listdir(repository_root)
    
    # 使用 ThreadPoolExecutor 并行处理每个代码库的删除操作
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(remove_lock_for_repo, repo): repo for repo in repos}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="删除 index.lock"):
            result = future.result()
            results.append(result)
            # 输出每个删除结果
            print(result)

def main():
    # 默认 repository 目录在 config.root_path 下
    repository_dir = os.path.join(config.root_path, "repository")
    global repository_root
    repository_root = repository_dir  # 在 remove_lock_for_repo 中使用
    remove_index_lock(repository_dir)

if __name__ == "__main__":
    main()