#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能: 并行检查 knowledge_base_all 中每个仓库的 has_file.json
      对应的 modified_file/<commit_hash> 目录下，
      before.*、after.*、diff.txt 文件是否存在且非空，且 before/after 唯一。
输出: 全局和每个仓库的符合/不符合统计，并列出所有不符合项及按错误原因聚合统计。
"""
import os
import sys
import json
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter

# 确保能导入 config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局配置
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base_all")
JSON_FILENAME       = "is_opt_keyword.json"
PATTERN_BEFORE      = "before.*"
PATTERN_AFTER       = "after.*"
MAX_WORKERS         = 128

def get_repository_list():
    if not os.path.isdir(KNOWLEDGE_BASE_PATH):
        print(f"ERROR: 知识库根目录不存在: {KNOWLEDGE_BASE_PATH}")
        sys.exit(1)
    return [
        name for name in os.listdir(KNOWLEDGE_BASE_PATH)
        if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, name))
    ]

def load_commits(repo_name):
    json_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, JSON_FILENAME)
    if not os.path.isfile(json_path):
        print(f"WARNING: 仓库 {repo_name} 缺少 {JSON_FILENAME}，跳过。")
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"WARNING: {json_path} 内容不是列表，跳过。")
            return []
        return data
    except Exception as e:
        print(f"WARNING: 读取或解析 {json_path} 失败: {e}")
        return []

def check_diff_files(repo_name, commit_hash):
    """
    检查 before/after/diff 三种文件是否都存在且非空。
    仅做存在性和大小检查，不尝试重算 diff.txt。
    返回 (is_valid, error_msg)
    """
    base_dir = os.path.join(
        KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash
    )
    if not os.path.isdir(base_dir):
        return False, "目录不存在"

    # 1) before
    before_list = glob.glob(os.path.join(base_dir, PATTERN_BEFORE))
    if len(before_list) != 1:
        return False, f"before 文件数量不对: 找到 {len(before_list)} 个"
    before_path = before_list[0]
    if os.path.getsize(before_path) == 0:
        return False, "before 文件内容为空"

    # 2) after
    after_list = glob.glob(os.path.join(base_dir, PATTERN_AFTER))
    if len(after_list) != 1:
        return False, f"after 文件数量不对: 找到 {len(after_list)} 个"
    after_path = after_list[0]
    if os.path.getsize(after_path) == 0:
        return False, "after 文件内容为空"

    # 3) diff
    diff_path = os.path.join(base_dir, "diff.txt")
    if not os.path.isfile(diff_path):
        return False, "diff.txt 不存在"
    if os.path.getsize(diff_path) == 0:
        return False, "diff.txt 内容为空"

    return True, ""

def process_repository(repo_name):
    commits = load_commits(repo_name)
    total = len(commits)
    ok    = 0
    nok   = 0
    failures = []  # list of (commit_hash, err)

    for item in commits:
        commit_hash = item.get('hash')
        if not commit_hash or not isinstance(commit_hash, str):
            nok += 1
            failures.append(("UNKNOWN_HASH", "缺少或无效的 hash 字段"))
            continue

        valid, err = check_diff_files(repo_name, commit_hash)
        if valid:
            ok += 1
        else:
            nok += 1
            failures.append((commit_hash, err))

    return {
        "repo": repo_name,
        "total": total,
        "ok": ok,
        "nok": nok,
        "failures": failures
    }

def main():
    repos = get_repository_list()
    print(f"找到 {len(repos)} 个仓库，使用 {MAX_WORKERS} 个进程并行检查…\n")

    global_total = 0
    global_ok    = 0
    global_nok   = 0
    all_failures = []  # list of (repo, hash, err)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_repo = {
            executor.submit(process_repository, repo): repo
            for repo in repos
        }

        for future in tqdm(as_completed(future_to_repo),
                           total=len(future_to_repo),
                           desc="检查仓库",
                           unit="repo"):
            repo = future_to_repo[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"[ERROR] 仓库 {repo} 处理异常: {e}")
                result = {"repo": repo, "total": 0, "ok": 0, "nok": 0, "failures": []}

            global_total += result["total"]
            global_ok    += result["ok"]
            global_nok   += result["nok"]
            for h, e in result["failures"]:
                all_failures.append((repo, h, e))

            print(f"[{repo}] total={result['total']}, ok={result['ok']}, nok={result['nok']}")

    # 全局统计
    print("\n" + "="*20 + " 全局统计 " + "="*20)
    print(f"全部 commits: {global_total}")
    print(f"符合要求    : {global_ok}")
    print(f"不符合要求  : {global_nok}")

    # 按错误原因聚合
    err_counter = Counter(err for _, _, err in all_failures)
    print("\n" + "="*10 + " 按错误原因统计 " + "="*10)
    for err, cnt in err_counter.most_common():
        print(f"{cnt} 个 commit => {err}")

    # 列出所有不符合项
    # if all_failures:
    #     print("\n以下 commit 不符合要求:")
    #     print("仓库 | commit_hash | 错误原因")
    #     for repo, h, err in all_failures:
    #         print(f"{repo} | {h} | {err}")

if __name__ == "__main__":
    main()