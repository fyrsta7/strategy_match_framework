#!/usr/bin/env python3
"""
统计所有commit的修改代码块数量
"""
import os
import json
import difflib
import concurrent.futures
from tqdm import tqdm
import threading
import tempfile
import shutil
import sys
import re
import time  # 添加time模块，用于进度刷新
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局配置变量
MAX_WORKERS = 128
INPUT_FILENAME = "line_block.json"
ENCODING_LIST = ['utf-8', 'latin-1', 'gbk', 'cp1252']
SKIP_PROCESSED = False  # 是否跳过已经处理过的 commit
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base_all")  # 新增全局变量

def count_code_blocks_from_git_diff(diff_content):
    """
    从 git diff 内容中统计连续的代码块数量
    Args:
        diff_content (str): git diff 的完整内容
    Returns:
        int: 修改的代码块数量
    """
    try:
        lines = diff_content.splitlines()
        code_blocks = 0
        # 新逻辑：将diff中的行按顺序处理，
        # 遇到连续的修改行(包含add/del连续交替)算同一个代码块，
        # 只有遇到上下文行(未修改行)时，代码块计数才加1。
        # 我们使用一个状态机：
        # state:
        #   'outside' - 在未修改状态
        #   'inside' - 正在修改一块内
        state = 'outside'
        # 当前代码块中，行是add还是del或者交替都认为是同一个代码块，只需区分修改和未修改
        # 只要当前行是添加或删除，即处于'inside'状态
        for line in lines:
            if line.startswith('@@'):
                # 代码块偏移指示符，不改变状态，继续下一行
                continue
            elif line.startswith(' '):
                # 上下文行，表示未修改
                if state == 'inside':
                    # 从修改转到未修改，代码块结束
                    code_blocks += 1
                    state = 'outside'
            elif line.startswith('+') and not line.startswith('+++'):
                # 添加行
                if state == 'outside':
                    state = 'inside'  # 新代码块开始
            elif line.startswith('-') and not line.startswith('---'):
                # 删除行
                if state == 'outside':
                    state = 'inside'  # 新代码块开始
            else:
                # 其他行，忽略，保持状态
                pass
        # 如果最后一块未闭合，计数加1
        if state == 'inside':
            code_blocks += 1
        return code_blocks
    except Exception as e:
        print(f"统计 git diff 代码块时出错: {e}")
        return 0

def count_code_blocks_from_file_diff(before_content, after_content):
    """
    从文件内容差异中统计连续的代码块数量
    Args:
        before_content (str): 修改前的文件内容
        after_content (str): 修改后的文件内容
    Returns:
        int: 修改的代码块数量
    """
    try:
        before_lines = before_content.splitlines(keepends=True)
        after_lines = after_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(before_lines, after_lines, lineterm=''))
        code_blocks = 0
        # 同上，根据统一diff的格式，逐行判断是否修改
        # 状态机方法，连续的add/del(可交替)记为同一代码块
        state = 'outside'
        for line in diff:
            if line.startswith('@@'):
                continue
            elif line.startswith(' '):
                # 上下文行，未修改
                if state == 'inside':
                    code_blocks += 1
                    state = 'outside'
            elif line.startswith('+') and not line.startswith('+++'):
                if state == 'outside':
                    state = 'inside'
            elif line.startswith('-') and not line.startswith('---'):
                if state == 'outside':
                    state = 'inside'
            else:
                # 忽略其他行（比如文件头部+++ ---）
                pass
        if state == 'inside':
            code_blocks += 1
        return code_blocks
    except Exception as e:
        print(f"统计文件差异代码块时出错: {e}")
        return 0

def safe_read_file(file_path):
    """
    安全读取文件，尝试多种编码
    Args:
        file_path (str): 文件路径
    Returns:
        str: 文件内容，读取失败返回空字符串
    """
    for encoding in ENCODING_LIST:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return ""

def process_single_commit(commit_data, repo_name):
    """
    处理单个 commit，计算其修改代码块数量
    Args:
        commit_data (dict): commit 数据
        repo_name (str): 仓库名称
    Returns:
        dict: 更新后的 commit 数据
    """
    if 'modified_code_blocks' in commit_data and SKIP_PROCESSED:
        return commit_data
    try:
        commit_hash = commit_data.get('hash', '')
        modified_files = commit_data.get('modified_files', [])
        if not commit_hash:
            commit_data.update({'modified_code_blocks': 0})
            return commit_data
        commit_dir = os.path.join(
            KNOWLEDGE_BASE_PATH,  # 使用全局变量
            repo_name,
            "modified_file",
            commit_hash
        )
        diff_path = os.path.join(commit_dir, "diff.txt")
        code_blocks = None
        # 方法1：优先尝试使用 diff.txt 文件
        if os.path.exists(diff_path):
            diff_content = safe_read_file(diff_path)
            if diff_content.strip():
                code_blocks = count_code_blocks_from_git_diff(diff_content)
        # 方法2：如果 diff.txt 不存在or为空，回退到使用 before/after 文件
        if code_blocks is None:
            if not modified_files or len(modified_files) == 0:
                commit_data.update({'modified_code_blocks': 0})
                return commit_data
            modified_file = modified_files[0]
            file_extension = os.path.splitext(modified_file)[1]
            before_path = os.path.join(commit_dir, f"before{file_extension}")
            after_path = os.path.join(commit_dir, f"after{file_extension}")
            if not os.path.exists(before_path) or not os.path.exists(after_path):
                print(f"before/after 文件不存在 - Commit: {commit_hash}")
                commit_data.update({'modified_code_blocks': 0})
                return commit_data
            before_content = safe_read_file(before_path)
            after_content = safe_read_file(after_path)
            if not before_content and not after_content:
                print(f"警告：before/after 文件内容均为空 - Commit: {commit_hash}")
            code_blocks = count_code_blocks_from_file_diff(before_content, after_content)
        # 如果两种方法都失败了
        if code_blocks is None:
            commit_data.update({'modified_code_blocks': 0})
            return commit_data
        commit_data['modified_code_blocks'] = code_blocks
        return commit_data
    except Exception as e:
        error_msg = f"处理 commit {commit_data.get('hash', 'unknown')} 时出错: {e}"
        print(error_msg)
        commit_data.update({'modified_code_blocks': 0})
        return commit_data

def process_repository(repo_name):
    """
    处理单个仓库的代码块统计
    Args:
        repo_name (str): 仓库名称
    Returns:
        tuple: (repo_name, processed_count, total_count, success_flag, error_msg)
    """
    repo_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name)  # 使用全局变量
    json_path = os.path.join(repo_path, INPUT_FILENAME)
    try:
        if not os.path.exists(json_path):
            return repo_name, 0, 0, False, f"文件不存在: {INPUT_FILENAME}"
        with open(json_path, 'r', encoding='utf-8') as f:
            commits_data = json.load(f)
        if not isinstance(commits_data, list):
            return repo_name, 0, 0, False, "JSON 文件格式错误，应为数组"
        original_count = len(commits_data)
        if original_count == 0:
            return repo_name, 0, 0, True, "没有需要处理的 commit"
        processed_commits = []
        processed_count = 0
        for commit_data in commits_data:
            try:
                updated_commit = process_single_commit(commit_data, repo_name)
                processed_commits.append(updated_commit)
                if 'modified_code_blocks' in updated_commit:
                    processed_count += 1
            except Exception as e:
                processed_commits.append(commit_data)
                print(f"处理 {repo_name} 中的 commit 时出错: {e}")
        # 原子性写入
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.json',
            dir=os.path.dirname(json_path),
            text=True
        )
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_f:
            json.dump(processed_commits, temp_f, indent=4, ensure_ascii=False)
        shutil.move(temp_path, json_path)
        return repo_name, processed_count, original_count, True, ""
    except Exception as e:
        return repo_name, 0, 0, False, f"处理异常: {str(e)}"

def analyze_code_blocks():
    """
    主函数：分析所有仓库中 commit 的修改代码块数量
    """
    if not os.path.exists(KNOWLEDGE_BASE_PATH):  # 使用全局变量
        print(f"错误: 知识库路径不存在: {KNOWLEDGE_BASE_PATH}")
        return
    
    repositories = [
        repo_name for repo_name in os.listdir(KNOWLEDGE_BASE_PATH)  # 使用全局变量
        if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, repo_name))
    ]
    # repositories = ["-eBPF-"]
    if not repositories:
        print("未找到任何代码仓库")
        return
    print(f"找到 {len(repositories)} 个代码库，正在统计总commit数量...")
    total_commits_count = 0
    repo_commit_counts = {}
    for repo_name in repositories:
        repo_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name)  # 使用全局变量
        json_path = os.path.join(repo_path, INPUT_FILENAME)
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    commits_data = json.load(f)
                    if isinstance(commits_data, list):
                        repo_commit_counts[repo_name] = len(commits_data)
                        total_commits_count += len(commits_data)
                    else:
                        repo_commit_counts[repo_name] = 0
            else:
                repo_commit_counts[repo_name] = 0
        except Exception as e:
            print(f"统计 {repo_name} commit数量时出错: {e}")
            repo_commit_counts[repo_name] = 0
    print(f"总计需要处理 {total_commits_count} 个 commit")
    print(f"使用 {MAX_WORKERS} 个并行线程进行处理")
    print(f"目标文件: {INPUT_FILENAME}")
    print("统计内容: 修改的代码块数量")
    total_success = 0
    total_processed_commits = 0
    total_commits = 0
    failed_repos = []
    stats_lock = threading.Lock()
    def update_stats(success, processed, total, repo_name, error_msg):
        nonlocal total_success, total_processed_commits, total_commits
        with stats_lock:
            if success:
                total_success += 1
                total_processed_commits += processed
                total_commits += total
            else:
                failed_repos.append((repo_name, error_msg))
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_repo = {
            executor.submit(process_repository, repo_name): repo_name
            for repo_name in repositories
        }
        with tqdm(total=total_commits_count, desc="处理Commit", unit="commit") as pbar:
            def refresh_progress():
                while not getattr(refresh_progress, 'stop', False):
                    pbar.refresh()
                    time.sleep(0.1)
            refresh_thread = threading.Thread(target=refresh_progress, daemon=True)
            refresh_thread.start()
            try:
                for future in concurrent.futures.as_completed(future_to_repo):
                    repo_name = future_to_repo[future]
                    try:
                        repo_name, processed_count, total_count, success, error_msg = future.result()
                        update_stats(success, processed_count, total_count, repo_name, error_msg)
                        pbar.update(repo_commit_counts.get(repo_name, 0))
                        pbar.set_postfix_str(f"仓库: {repo_name} | 成功: {processed_count}/{total_count}")
                        if not success:
                            tqdm.write(f"[失败] {repo_name}: {error_msg}")
                    except Exception as e:
                        pbar.update(repo_commit_counts.get(repo_name, 0))
                        tqdm.write(f"[异常] {repo_name}: {str(e)}")
                        update_stats(False, 0, 0, repo_name, f"处理异常: {str(e)}")
            finally:
                refresh_progress.stop = True
                refresh_thread.join(timeout=1)
    print(f"\n分析完成!")
    print(f"成功处理: {total_success} 个代码库")
    print(f"总计处理: {total_processed_commits} 个 commit (共 {total_commits} 个)")
    if failed_repos:
        print(f"\n失败的代码库 ({len(failed_repos)} 个):")
        for repo_name, error_msg in failed_repos[:10]:
            print(f"  - {repo_name}: {error_msg}")
        if len(failed_repos) > 10:
            print(f"  ... 还有 {len(failed_repos) - 10} 个失败案例")
    success_rate = (total_success / len(repositories)) * 100 if repositories else 0
    print(f"\n成功率: {success_rate:.1f}%")

if __name__ == "__main__":
    analyze_code_blocks()