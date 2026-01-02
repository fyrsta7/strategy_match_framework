import os
import glob
import shutil
import traceback
from collections import defaultdict, Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

# 加载上层目录中的 config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 路径配置
kb1_path = os.path.join(config.root_path, "knowledge_base_all")
kb2_path = os.path.join(config.root_path, "knowledge_base_all_gpu")
MAX_WORKERS = 32


def process_commit(args):
    repo_name, commit_hash = args
    result = {
        'repo': repo_name,
        'hash': commit_hash,
        'status': 'success',
        'reason': None
    }

    try:
        commit_kb1_path = os.path.join(kb1_path, repo_name, "modified_file", commit_hash)
        commit_kb2_path = os.path.join(kb2_path, repo_name, "modified_file", commit_hash)

        if not os.path.isdir(commit_kb2_path):
            result['status'] = 'failure'
            result['reason'] = 'missing_target_commit_path'
            return result

        before_files = glob.glob(os.path.join(commit_kb2_path, "before.*"))
        after_files = glob.glob(os.path.join(commit_kb2_path, "after.*"))

        if len(before_files) == 0:
            result['status'] = 'failure'
            result['reason'] = 'missing_before_file'
            return result
        if len(before_files) > 1:
            result['status'] = 'failure'
            result['reason'] = 'multiple_before_files'
            return result
        if len(after_files) == 0:
            result['status'] = 'failure'
            result['reason'] = 'missing_after_file'
            return result
        if len(after_files) > 1:
            result['status'] = 'failure'
            result['reason'] = 'multiple_after_files'
            return result

        # 执行复制
        shutil.copy2(before_files[0], commit_kb1_path)
        shutil.copy2(after_files[0], commit_kb1_path)

    except Exception:
        traceback.print_exc()
        result['status'] = 'failure'
        result['reason'] = 'unknown_error'

    return result


def main():
    all_commits = []

    for repo_name in os.listdir(kb1_path):
        repo_kb1_path = os.path.join(kb1_path, repo_name, "modified_file")
        if not os.path.isdir(repo_kb1_path):
            continue

        for commit_hash in os.listdir(repo_kb1_path):
            all_commits.append((repo_name, commit_hash))

    total_commits = len(all_commits)
    success_count = 0
    failure_counter = Counter()

    print(f"Total commits to process: {total_commits}")

    # 使用多进程处理
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_commit, args): args for args in all_commits}

        with tqdm(total=total_commits, desc="Processing commits") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                repo = result['repo']
                commit_hash = result['hash']
                status = result['status']
                reason = result['reason']

                if status == 'success':
                    success_count += 1
                else:
                    failure_counter[reason] += 1
                    print(f"[FAILED] {repo}/{commit_hash} - Reason: {reason}")

    # 打印统计信息
    print("\n====== SUMMARY ======")
    print(f"Total commits processed: {total_commits}")
    print(f"Commits with successful file copy: {success_count}")
    print(f"Failed commits: {total_commits - success_count}")
    print("Failure reasons:")
    for reason, count in failure_counter.items():
        print(f" - {reason}: {count}")


if __name__ == "__main__":
    main()
