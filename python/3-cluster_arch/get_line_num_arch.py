import os
import json
import glob
import difflib
import re
import sys
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
SUMMARY_JSON_FILENAME = "summary_arch.json"
REGENERATE_EXISTING = True
MAX_WORKER_THREADS = 256
ENABLE_PARALLEL_PROCESSING = True

# 全局统计锁
stats_lock = Lock()
global_stats = {
    'repositories_total': 0,
    'repositories_success': 0,
    'repositories_failed': 0,
    'commits_total': 0,
    'commits_success': 0,
    'commits_skipped': 0,
    'commits_failed': 0
}

def find_file_with_pattern(directory: str, pattern: str) -> Optional[str]:
    """查找符合模式的文件，要求必须唯一匹配"""
    if not os.path.exists(directory):
        return None
    
    pattern_files = glob.glob(os.path.join(directory, pattern))
    
    if len(pattern_files) == 0:
        return None
    elif len(pattern_files) == 1:
        return pattern_files[0]
    else:
        print(f"错误: 模式 '{pattern}' 匹配到多个文件: {pattern_files}")
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
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        return [line.rstrip('\r\n') + '\n' for line in lines]
    except Exception:
        return []

def calculate_file_diff(before_lines: List[str], after_lines: List[str]) -> Tuple[Optional[int], Optional[int], float]:
    """
    使用difflib计算文件diff，返回修改的起始和结束行号
    返回: (start_line, end_line, confidence)
    """
    if not before_lines or not after_lines:
        return None, None, 0.0
    
    try:
        diff_lines = list(difflib.unified_diff(
            before_lines, 
            after_lines, 
            lineterm='',
            n=0
        ))
        
        if not diff_lines:
            return None, None, 0.0
        
        hunk_info = []
        for line in diff_lines:
            if line.startswith('@@'):
                match = re.match(r'@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1
                    
                    hunk_info.append({
                        'old_start': old_start,
                        'old_count': old_count,
                        'new_start': new_start,
                        'new_count': new_count
                    })
        
        if not hunk_info:
            return None, None, 0.0
        
        start_line = min(hunk['old_start'] for hunk in hunk_info)
        
        max_end = 0
        for hunk in hunk_info:
            if hunk['old_count'] > 0:
                end = hunk['old_start'] + hunk['old_count'] - 1
            else:
                end = hunk['old_start']
            max_end = max(max_end, end)
        
        end_line = max_end
        
        if start_line <= 0 or end_line <= 0 or start_line > len(before_lines):
            return None, None, 0.0
        
        end_line = min(end_line, len(before_lines))
        
        if start_line > end_line:
            end_line = start_line
        
        confidence = min(1.0, 0.8 + 0.1 * min(len(hunk_info), 2))
        
        return start_line, end_line, confidence
        
    except Exception as e:
        return None, None, 0.0

def process_commit_diff(commit, repo_name):
    """处理单个commit的diff计算"""
    result = {
        'file_start_line': None,
        'file_end_line': None,
        'confidence': 0.0,
        'status': 'failed',
        'error': None
    }
    
    try:
        if not REGENERATE_EXISTING:
            if 'file_start_line' in commit and 'file_end_line' in commit:
                if commit['file_start_line'] is not None and commit['file_end_line'] is not None:
                    result.update({
                        'file_start_line': commit['file_start_line'],
                        'file_end_line': commit['file_end_line'],
                        'status': 'skipped',
                        'confidence': 1.0
                    })
                    return result
        
        commit_hash = commit.get('hash')
        
        if not repo_name or not commit_hash:
            result['error'] = 'Missing repository_name or hash'
            return result
        
        commit_dir = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash)
        
        if not os.path.exists(commit_dir):
            result['error'] = f'Commit directory not found'
            return result
        
        before_file = find_file_with_pattern(commit_dir, "before.*")
        after_file = find_file_with_pattern(commit_dir, "after.*")
        
        if not before_file or not after_file:
            result['error'] = 'Before or after file not found'
            return result
        
        before_lines = read_file_with_encoding(before_file)
        after_lines = read_file_with_encoding(after_file)
        
        if not before_lines or not after_lines:
            result['error'] = 'File is empty or unreadable'
            return result
        
        start_line, end_line, confidence = calculate_file_diff(before_lines, after_lines)
        
        if start_line is None or end_line is None:
            result['error'] = 'Failed to calculate diff'
            return result
        
        result.update({
            'file_start_line': start_line,
            'file_end_line': end_line,
            'confidence': confidence,
            'status': 'success'
        })
        
        return result
        
    except Exception as e:
        result['error'] = f'Processing error: {str(e)}'
        return result

def process_single_repository(repo):
    """处理单个代码库"""
    repo_name = repo.get('name_long', repo.get('name', ''))
    
    result = {
        'repository_name': repo_name,
        'status': 'failed',
        'commits_total': 0,
        'commits_success': 0,
        'commits_skipped': 0,
        'commits_failed': 0,
        'error': None
    }
    
    try:
        json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, SUMMARY_JSON_FILENAME)
        
        if not os.path.exists(json_file_path):
            result['error'] = 'JSON file not found'
            return result
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
        
        if not commits:
            result['status'] = 'success'
            result['error'] = 'No commits found'
            return result
        
        result['commits_total'] = len(commits)
        
        for commit in commits:
            commit_result = process_commit_diff(commit, repo_name)
            
            if commit_result['status'] == 'success':
                commit['file_start_line'] = commit_result['file_start_line']
                commit['file_end_line'] = commit_result['file_end_line']
                result['commits_success'] += 1
            elif commit_result['status'] == 'skipped':
                result['commits_skipped'] += 1
            else:
                result['commits_failed'] += 1
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(commits, f, ensure_ascii=False, indent=2)
        
        result['status'] = 'success'
        
    except Exception as e:
        result['error'] = f'Repository processing error: {str(e)}'
    
    return result

def main():
    """主函数"""
    print("=" * 80)
    print("计算体系结构相关策略的代码行号")
    print("=" * 80)
    print(f"知识库路径: {KNOWLEDGE_BASE_PATH}")
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"目标JSON文件: {SUMMARY_JSON_FILENAME}")
    print(f"重新处理已有字段: {REGENERATE_EXISTING}")
    print("-" * 80)
    
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件不存在 - {REPO_LIST_FILE}")
        return
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repositories = json.load(f)
    
    print(f"发现 {len(repositories)} 个代码库")
    
    if not repositories:
        print("没有代码库需要处理")
        return
    
    print(f"\n开始处理...")
    
    start_time = time.time()
    
    if ENABLE_PARALLEL_PROCESSING and len(repositories) > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
            future_to_repo = {
                executor.submit(process_single_repository, repo): repo 
                for repo in repositories
            }
            
            with tqdm(total=len(repositories), desc="处理代码库", unit="repo") as pbar:
                for future in as_completed(future_to_repo):
                    repo_info = future_to_repo[future]
                    repo_name = repo_info.get('name_long', repo_info.get('name', ''))
                    
                    try:
                        result = future.result()
                        
                        with stats_lock:
                            global_stats['repositories_total'] += 1
                            if result['status'] == 'success':
                                global_stats['repositories_success'] += 1
                                global_stats['commits_total'] += result['commits_total']
                                global_stats['commits_success'] += result['commits_success']
                                global_stats['commits_skipped'] += result['commits_skipped']
                                global_stats['commits_failed'] += result['commits_failed']
                            else:
                                global_stats['repositories_failed'] += 1
                        
                    except Exception as e:
                        print(f"\n异常: {repo_name} - {str(e)}")
                        with stats_lock:
                            global_stats['repositories_total'] += 1
                            global_stats['repositories_failed'] += 1
                    
                    pbar.update(1)
    else:
        for repo in tqdm(repositories, desc="处理代码库", unit="repo"):
            result = process_single_repository(repo)
            
            global_stats['repositories_total'] += 1
            if result['status'] == 'success':
                global_stats['repositories_success'] += 1
                global_stats['commits_total'] += result['commits_total']
                global_stats['commits_success'] += result['commits_success']
                global_stats['commits_skipped'] += result['commits_skipped']
                global_stats['commits_failed'] += result['commits_failed']
            else:
                global_stats['repositories_failed'] += 1
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)
    print(f"总处理时间: {total_time:.1f}s")
    print(f"代码库统计:")
    print(f"  总计: {global_stats['repositories_total']}")
    print(f"  成功: {global_stats['repositories_success']}")
    print(f"  失败: {global_stats['repositories_failed']}")
    print(f"Commit统计:")
    print(f"  总计: {global_stats['commits_total']}")
    print(f"  成功: {global_stats['commits_success']}")
    print(f"  跳过: {global_stats['commits_skipped']}")
    print(f"  失败: {global_stats['commits_failed']}")
    
    if global_stats['commits_total'] > 0:
        success_rate = (global_stats['commits_success'] / global_stats['commits_total']) * 100
        print(f"Commit成功率: {success_rate:.1f}%")

if __name__ == "__main__":
    main()

