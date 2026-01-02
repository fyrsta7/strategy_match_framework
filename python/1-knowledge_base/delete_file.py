import os
import json
import concurrent.futures
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局配置
knowledge_base_path = os.path.join(config.root_path, "knowledge_base_all")
commits_file_name = "diff.json"
files_to_delete = ["before_func.txt", "after_func.txt"]
max_workers = 8


def process_repository(repo_name):
    """
    处理单个仓库：读取 diff.json，遍历每个 commit，删除指定的文件。
    返回 (repo_name, total_commits, total_deleted, status)
    """
    repo_root = os.path.join(knowledge_base_path, repo_name)
    diff_path = os.path.join(repo_root, commits_file_name)

    if not os.path.exists(diff_path):
        return repo_name, 0, 0, "未找到 diff.json"

    try:
        with open(diff_path, "r", encoding="utf-8") as f:
            commits = json.load(f)
    except Exception as e:
        return repo_name, 0, 0, f"读取 diff.json 错误: {e}"

    total_commits = len(commits)
    total_deleted = 0
    errors = []

    for c in commits:
        commit_hash = c.get("hash", "").strip()
        if not commit_hash:
            errors.append("缺少 hash 字段")
            continue

        mod_dir = os.path.join(repo_root, "modified_file", commit_hash)
        if not os.path.isdir(mod_dir):
            errors.append(f"{commit_hash}: 找不到 modified_file 目录")
            continue

        for fname in files_to_delete:
            fp = os.path.join(mod_dir, fname)
            if os.path.exists(fp):
                if os.path.isfile(fp):
                    try:
                        os.remove(fp)
                        total_deleted += 1
                    except Exception as e:
                        errors.append(f"{commit_hash}/{fname}: 删除失败 ({e})")
                else:
                    errors.append(f"{commit_hash}/{fname}: 不是文件，无法删除")
            # 文件不存在则忽略

    if not errors:
        status = "成功"
    else:
        # 可以只保留前几条错误，防止过长
        preview = errors if len(errors) <= 5 else errors[:5] + ["..."]
        status = f"部分错误: {preview}"

    return repo_name, total_commits, total_deleted, status


def main():
    # 列出所有仓库
    try:
        repos = [
            d for d in os.listdir(knowledge_base_path)
            if os.path.isdir(os.path.join(knowledge_base_path, d))
        ]
    except Exception as e:
        print(f"无法列出目录 {knowledge_base_path}: {e}")
        return []

    print(f"开始并行处理 {len(repos)} 个代码库...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_repo = {
            executor.submit(process_repository, repo): repo
            for repo in repos
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_repo),
            total=len(repos),
            desc="处理仓库进度"
        ):
            repo = future_to_repo[future]
            try:
                res = future.result()
            except Exception as e:
                res = (repo, 0, 0, f"线程异常: {e}")
            results.append(res)

    # 汇总统计
    success_count = sum(1 for _, _, _, st in results if st == "成功")
    total_commits = sum(tc for _, tc, _, st in results if st == "成功")
    total_deleted = sum(td for _, _, td, st in results if st == "成功")

    print("\n" + "-" * 60)
    print("处理完成！结果统计：")
    print(f"成功处理仓库: {success_count}/{len(repos)}")
    print(f"总提交数 (成功仓库): {total_commits}")
    print(f"总删除文件数 (成功仓库): {total_deleted}")

    print("\n各仓库明细：")
    for repo, tc, td, st in sorted(results, key=lambda x: x[0]):
        print(f"{repo}: 提交 {tc} 个, 删除 {td} 个, 状态: {st}")

    print("-" * 60)
    return results


if __name__ == "__main__":
    main()