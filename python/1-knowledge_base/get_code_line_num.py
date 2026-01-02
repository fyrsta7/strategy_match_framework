#!/usr/bin/env python3
"""
统计所有commit的总修改行数（添加、删除、总修改、净变化）
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
import time  # 添加缺失的import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局配置变量
MAX_WORKERS = 128
INPUT_FILENAME = "line_block.json"
ENCODING_LIST = ['utf-8', 'latin-1', 'gbk', 'cp1252']
SKIP_PROCESSED = False  # 是否跳过已经处理过的 commit
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base_all")  # 新增全局变量

def parse_git_diff(diff_content):
    """
    解析 git diff 内容，统计修改行数
    
    Args:
        diff_content (str): git diff 的完整内容
    
    Returns:
        dict: 包含变更统计信息的字典
    """
    try:
        lines = diff_content.splitlines()
        added_lines = 0
        deleted_lines = 0
        
        for line in lines:
            # 统计添加的行（以 + 开头，但不包括 +++ 文件头）
            if line.startswith('+') and not line.startswith('+++'):
                added_lines += 1
            # 统计删除的行（以 - 开头，但不包括 --- 文件头）
            elif line.startswith('-') and not line.startswith('---'):
                deleted_lines += 1
        
        total_changes = added_lines + deleted_lines
        net_change = added_lines - deleted_lines
        
        return {
            'added_lines': added_lines,
            'deleted_lines': deleted_lines,
            'total_changed_lines': total_changes,
            'net_line_change': net_change
        }
    
    except Exception as e:
        print(f"解析 git diff 时出错: {e}")
        return {
            'added_lines': 0,
            'deleted_lines': 0,
            'total_changed_lines': 0,
            'net_line_change': 0
        }

def count_file_changes(before_content, after_content):
    """
    统计两个文件内容之间的变更行数
    
    Args:
        before_content (str): 修改前的文件内容
        after_content (str): 修改后的文件内容
    
    Returns:
        dict: 包含变更统计信息的字典
    """
    try:
        before_lines = before_content.splitlines(keepends=True)
        after_lines = after_content.splitlines(keepends=True)
        
        # 使用 difflib 生成统一差异
        diff = list(difflib.unified_diff(before_lines, after_lines, lineterm=''))
        
        added_lines = 0
        deleted_lines = 0
        
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                added_lines += 1
            elif line.startswith('-') and not line.startswith('---'):
                deleted_lines += 1
        
        total_changes = added_lines + deleted_lines
        net_change = len(after_lines) - len(before_lines)
        
        return {
            'added_lines': added_lines,
            'deleted_lines': deleted_lines,
            'total_changed_lines': total_changes,
            'net_line_change': net_change
        }
    
    except Exception as e:
        print(f"计算文件变更时出错: {e}")
        return {
            'added_lines': 0,
            'deleted_lines': 0,
            'total_changed_lines': 0,
            'net_line_change': 0
        }

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
    
    # 所有编码都失败，返回空内容
    return ""

def process_single_commit(commit_data, repo_name):
    """
    处理单个 commit，计算其代码修改行数
    
    Args:
        commit_data (dict): commit 数据
        repo_name (str): 仓库名称
    
    Returns:
        dict: 更新后的 commit 数据
    """
    # 如果已经有 total_changed_lines 字段，则跳过
    if 'total_changed_lines' in commit_data and SKIP_PROCESSED:
        return commit_data
    
    try:
        # 获取commit hash和修改的文件信息
        commit_hash = commit_data.get('hash', '')
        modified_files = commit_data.get('modified_files', [])
        
        if not commit_hash:
            # 没有commit hash，设置默认值
            commit_data.update({
                'added_lines': 0,
                'deleted_lines': 0,
                'total_changed_lines': 0,
                'net_line_change': 0
            })
            return commit_data
        
        # 构建文件所在目录路径 - 使用全局变量
        commit_dir = os.path.join(
            KNOWLEDGE_BASE_PATH,  # 使用全局变量
            repo_name, 
            "modified_file", 
            commit_hash
        )
        
        diff_path = os.path.join(commit_dir, "diff.txt")
        changes = None
        
        # 方法1：优先尝试使用 diff.txt 文件
        if os.path.exists(diff_path):
            diff_content = safe_read_file(diff_path)
            if diff_content.strip():
                changes = parse_git_diff(diff_content)
        
        # 方法2：如果 diff.txt 不存在或为空，回退到使用 before/after 文件
        if changes is None:
            if not modified_files or len(modified_files) == 0:
                commit_data.update({
                    'added_lines': 0,
                    'deleted_lines': 0,
                    'total_changed_lines': 0,
                    'net_line_change': 0
                })
                return commit_data
            
            modified_file = modified_files[0]
            file_extension = os.path.splitext(modified_file)[1]
            before_path = os.path.join(commit_dir, f"before{file_extension}")
            after_path = os.path.join(commit_dir, f"after{file_extension}")
            
            if not os.path.exists(before_path) or not os.path.exists(after_path):
                print(f"before/after 文件不存在 - Commit: {commit_hash}")
                commit_data.update({
                    'added_lines': 0,
                    'deleted_lines': 0,
                    'total_changed_lines': 0,
                    'net_line_change': 0
                })
                return commit_data
            
            before_content = safe_read_file(before_path)
            after_content = safe_read_file(after_path)
            
            if not before_content and not after_content:
                print(f"警告：before/after 文件内容均为空 - Commit: {commit_hash}")
            
            changes = count_file_changes(before_content, after_content)
        
        # 如果两种方法都失败了
        if changes is None:
            commit_data.update({
                'added_lines': 0,
                'deleted_lines': 0,
                'total_changed_lines': 0,
                'net_line_change': 0
            })
            return commit_data
        
        # 更新行数相关的四个字段
        commit_data.update(changes)
        
        return commit_data
    
    except Exception as e:
        error_msg = f"处理 commit {commit_data.get('hash', 'unknown')} 时出错: {e}"
        print(error_msg)
        commit_data.update({
            'added_lines': 0,
            'deleted_lines': 0,
            'total_changed_lines': 0,
            'net_line_change': 0
        })
        return commit_data

def process_repository(repo_name):
    """
    处理单个仓库的修改行数统计
    
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
                
                if 'total_changed_lines' in updated_commit:
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

def analyze_line_changes():
    """
    主函数：分析所有仓库中 commit 的代码修改行数
    """
    # 使用全局变量
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"错误: 知识库路径不存在: {KNOWLEDGE_BASE_PATH}")
        return
    
    repositories = [
        repo_name for repo_name in os.listdir(KNOWLEDGE_BASE_PATH)
        if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, repo_name))
    ]
    # repositories = ["1brc"]
    
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
    print("统计内容: 添加行数、删除行数、总修改行数、净行数变化")
    
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
    analyze_line_changes()