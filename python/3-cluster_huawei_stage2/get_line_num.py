import os
import json
import glob
import difflib
import re
import sys
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")
INPUT_FILE = "summary_huawei.json"
OUTPUT_FILE = "summary_huawei.json"
MAX_REPO_WORKERS = 16  # 代码库级别并行数
MAX_WORKERS = 128  # commit 级别并行数
REGENERATE_EXISTING = False  # 是否重新处理已有字段

# 全局统计锁
stats_lock = Lock()
global_stats = {
    'total_repos': 0,
    'success_repos': 0,
    'total_commits': 0,
    'success_commits': 0,
    'skipped_commits': 0,
    'failed_commits': 0
}

def find_file_with_pattern(directory: str, pattern: str) -> Optional[str]:
    """查找符合精确模式的文件"""
    if not os.path.exists(directory):
        return None
    
    # 尝试精确匹配
    exact_pattern = os.path.join(directory, pattern)
    if os.path.exists(exact_pattern):
        return exact_pattern
    
    # 使用精确的点分隔模式
    pattern_files = glob.glob(os.path.join(directory, pattern + '.*'))
    
    if len(pattern_files) == 1:
        return pattern_files[0]
    
    return None

def read_file_with_encoding(file_path: str) -> List[str]:
    """使用多种编码尝试读取文件"""
    if not file_path or not os.path.exists(file_path):
        return []
    
    encodings = ['utf-8', 'gbk', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            return [line.rstrip('\r\n') + '\n' for line in lines]
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    # 最后尝试忽略错误
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        return [line.rstrip('\r\n') + '\n' for line in lines]
    except Exception:
        return []

def normalize_file_content(lines: List[str]) -> List[str]:
    """标准化文件内容"""
    if not lines:
        return []
    
    # 移除最后的空行
    while lines and lines[-1].strip() == '':
        lines.pop()
    
    # 确保最后一行有换行符
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    
    return lines

def calculate_file_diff(before_lines: List[str], after_lines: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """使用difflib计算文件diff，返回修改的起始和结束行号"""
    if not before_lines or not after_lines:
        return None, None
    
    try:
        diff_lines = list(difflib.unified_diff(
            before_lines, 
            after_lines, 
            lineterm='',
            n=0
        ))
        
        if not diff_lines:
            return None, None
        
        # 解析hunk信息
        hunk_info = []
        for line in diff_lines:
            if line.startswith('@@'):
                match = re.match(r'@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    hunk_info.append({
                        'old_start': old_start,
                        'old_count': old_count
                    })
        
        if not hunk_info:
            return None, None
        
        # 计算修改范围
        start_line = min(hunk['old_start'] for hunk in hunk_info)
        
        max_end = 0
        for hunk in hunk_info:
            if hunk['old_count'] > 0:
                end = hunk['old_start'] + hunk['old_count'] - 1
            else:
                end = hunk['old_start']
            max_end = max(max_end, end)
        
        end_line = max_end
        
        # 验证结果合理性
        if start_line <= 0 or end_line <= 0 or start_line > len(before_lines):
            return None, None
        
        end_line = min(end_line, len(before_lines))
        
        if start_line > end_line:
            end_line = start_line
        
        return start_line, end_line
        
    except Exception:
        return None, None

def process_commit(commit, repo_name):
    """处理单个commit的diff计算"""
    # 检查是否需要重新处理
    if not REGENERATE_EXISTING:
        if 'file_start_line' in commit and 'file_end_line' in commit:
            if commit['file_start_line'] is not None and commit['file_end_line'] is not None:
                return {'status': 'skipped'}
    
    # 获取必要信息
    commit_hash = commit.get('hash')
    if not commit_hash:
        return {'status': 'failed', 'error': 'Missing hash'}
    
    # 构建文件路径
    commit_dir = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash)
    
    if not os.path.exists(commit_dir):
        return {'status': 'failed', 'error': 'Commit directory not found'}
    
    # 查找before和after文件
    before_file = find_file_with_pattern(commit_dir, "before")
    after_file = find_file_with_pattern(commit_dir, "after")
    
    if not before_file or not after_file:
        return {'status': 'failed', 'error': 'Before or after file not found'}
    
    # 读取文件内容
    before_lines = read_file_with_encoding(before_file)
    after_lines = read_file_with_encoding(after_file)
    
    if not before_lines or not after_lines:
        return {'status': 'failed', 'error': 'File empty or unreadable'}
    
    # 标准化文件内容
    before_lines = normalize_file_content(before_lines)
    after_lines = normalize_file_content(after_lines)
    
    # 计算diff
    start_line, end_line = calculate_file_diff(before_lines, after_lines)
    
    if start_line is None or end_line is None:
        return {'status': 'failed', 'error': 'Failed to calculate diff'}
    
    return {
        'status': 'success',
        'file_start_line': start_line,
        'file_end_line': end_line
    }

def process_repository(repo_name):
    """处理单个代码库"""
    json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, INPUT_FILE)
    
    if not os.path.exists(json_file_path):
        return {'repo_name': repo_name, 'status': 'file_not_found', 'stats': {}}
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
    except Exception as e:
        return {'repo_name': repo_name, 'status': 'read_error', 'error': str(e), 'stats': {}}
    
    if not commits:
        return {'repo_name': repo_name, 'status': 'empty', 'stats': {}}
    
    stats = {'total': len(commits), 'success': 0, 'skipped': 0, 'failed': 0}
    
    # 处理每个commit
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {executor.submit(process_commit, commit, repo_name): i for i, commit in enumerate(commits)}
        
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                result = future.result()
                
                if result['status'] == 'success':
                    commits[i]['file_start_line'] = result['file_start_line']
                    commits[i]['file_end_line'] = result['file_end_line']
                    stats['success'] += 1
                elif result['status'] == 'skipped':
                    stats['skipped'] += 1
                else:
                    stats['failed'] += 1
            except Exception:
                stats['failed'] += 1
    
    # 保存结果
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(commits, f, ensure_ascii=False, indent=2)
        return {'repo_name': repo_name, 'status': 'success', 'stats': stats}
    except Exception as e:
        return {'repo_name': repo_name, 'status': 'write_error', 'error': str(e), 'stats': stats}

def main():
    """主函数"""
    print("=" * 80)
    print("计算被修改代码片段的行号")
    print("=" * 80)
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"知识库路径: {KNOWLEDGE_BASE_PATH}")
    print(f"输入/输出文件: {INPUT_FILE}")
    print(f"代码库级别并行数: {MAX_REPO_WORKERS}")
    print(f"Commit级别并行数: {MAX_WORKERS}")
    print("-" * 80)
    
    # 读取代码库列表
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件不存在 - {REPO_LIST_FILE}")
        return
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repo_list = json.load(f)
    
    repositories = []
    for repo in repo_list:
        repo_name = repo.get('name_long') or repo.get('name')
        if repo_name:
            # 只处理存在 summary_huawei.json 的代码库
            json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, INPUT_FILE)
            if os.path.exists(json_file_path):
                repositories.append(repo_name)
    
    print(f"发现 {len(repositories)} 个需要处理的代码库")
    
    if not repositories:
        print("没有需要处理的代码库")
        return
    
    # 处理所有代码库
    with ThreadPoolExecutor(max_workers=MAX_REPO_WORKERS) as executor:
        future_to_repo = {executor.submit(process_repository, repo): repo for repo in repositories}
        
        pbar = tqdm(total=len(repositories), desc="处理代码库", unit="repo")
        
        for future in as_completed(future_to_repo):
            try:
                result = future.result()
                
                with stats_lock:
                    global_stats['total_repos'] += 1
                    if result['status'] == 'success':
                        global_stats['success_repos'] += 1
                        global_stats['total_commits'] += result['stats']['total']
                        global_stats['success_commits'] += result['stats']['success']
                        global_stats['skipped_commits'] += result['stats']['skipped']
                        global_stats['failed_commits'] += result['stats']['failed']
                
                pbar.set_postfix({
                    "成功": global_stats['success_commits'],
                    "失败": global_stats['failed_commits']
                })
            except Exception:
                pass
            
            pbar.update(1)
        
        pbar.close()
    
    # 输出统计
    print("\n" + "=" * 80)
    print("处理完成 - 统计信息")
    print("=" * 80)
    print(f"代码库统计:")
    print(f"  总数: {global_stats['total_repos']}")
    print(f"  成功: {global_stats['success_repos']}")
    print(f"Commit统计:")
    print(f"  总数: {global_stats['total_commits']}")
    print(f"  成功: {global_stats['success_commits']}")
    print(f"  跳过: {global_stats['skipped_commits']}")
    print(f"  失败: {global_stats['failed_commits']}")
    
    if global_stats['total_commits'] > 0:
        success_rate = (global_stats['success_commits'] / global_stats['total_commits']) * 100
        print(f"成功率: {success_rate:.1f}%")
    
    print("\n处理完成！")

if __name__ == "__main__":
    main()

