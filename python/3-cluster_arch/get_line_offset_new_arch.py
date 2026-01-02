import os
import json
import glob
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
REGENERATE_EXISTING = True  # 是否重新处理已有字段的commit（True表示重新处理，False表示使用已有结果）
MAX_WORKER_THREADS = 16  # 代码库级别并行
MAX_COMMIT_WORKER_THREADS = 16  # commit级别并行
ENABLE_PARALLEL_PROCESSING = True  # 是否启用代码库级别并行处理
ENABLE_COMMIT_PARALLEL_PROCESSING = True  # 是否启用commit级别并行处理

# 全局统计锁
stats_lock = Lock()
global_stats = {
    'repositories_total': 0,
    'repositories_success': 0,
    'repositories_failed': 0,
    'commits_total': 0,
    'commits_success': 0,
    'commits_partial': 0,
    'commits_skipped': 0,
    'commits_failed': 0
}

def find_unique_file_with_pattern(directory: str, pattern: str) -> Optional[str]:
    """查找符合模式的文件，要求必须唯一匹配"""
    if not os.path.exists(directory):
        return None
    
    pattern_files = glob.glob(os.path.join(directory, pattern))
    if len(pattern_files) == 0:
        return None
    elif len(pattern_files) == 1:
        return pattern_files[0]
    else:
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
            return lines
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        return lines
    except Exception:
        return []

def count_file_lines(file_path: str) -> int:
    """统计文件总行数"""
    lines = read_file_with_encoding(file_path)
    return len(lines)

def normalize_line_for_matching(line: str) -> str:
    """标准化行内容用于匹配"""
    return line.strip()

def find_line_in_file(target_line: str, file_lines: List[str], start_from: int = 0) -> Optional[int]:
    """在文件中查找指定行，返回行号（1-based）"""
    normalized_target = normalize_line_for_matching(target_line)
    
    for i in range(start_from, len(file_lines)):
        if normalize_line_for_matching(file_lines[i]) == normalized_target:
            return i + 1
    
    return None

def calculate_line_offset(before_file_path: str, before_func_file_path: str) -> Tuple[Optional[int], str]:
    """
    计算before文件和before_func文件之间的行号偏移
    返回: (offset, status)
    offset: before文件中函数起始行号 - 1（因为before_func第一行对应的实际行号）
    status: 计算状态（success, failed, etc.）
    """
    try:
        before_lines = read_file_with_encoding(before_file_path)
        before_func_lines = read_file_with_encoding(before_func_file_path)
        
        if not before_lines or not before_func_lines:
            return None, "empty_file"
        
        # 取before_func的第一行（非空）
        first_func_line = None
        for line in before_func_lines:
            if line.strip():
                first_func_line = line
                break
        
        if not first_func_line:
            return None, "no_content_in_func_file"
        
        # 在before文件中查找这一行
        line_num = find_line_in_file(first_func_line, before_lines)
        
        if line_num is None:
            return None, "line_not_found"
        
        # offset = before文件中的行号 - 1（因为before_func第1行对应before中的line_num行）
        offset = line_num - 1
        
        return offset, "success"
        
    except Exception as e:
        return None, f"exception: {str(e)}"

def process_single_commit(commit, repo_name):
    """处理单个commit"""
    result = {
        'line_offset': None,
        'func_start_line': None,
        'func_end_line': None,
        'before_func_total_lines': None,
        'status': 'failed',
        'error': None
    }
    
    try:
        # 检查是否需要重新处理
        if not REGENERATE_EXISTING:
            has_all_fields = all(k in commit for k in ['line_offset', 'func_start_line', 'func_end_line', 'before_func_total_lines'])
            if has_all_fields:
                result.update({
                    'line_offset': commit['line_offset'],
                    'func_start_line': commit['func_start_line'],
                    'func_end_line': commit['func_end_line'],
                    'before_func_total_lines': commit['before_func_total_lines'],
                    'status': 'skipped'
                })
                return result
        
        commit_hash = commit.get('hash')
        if not commit_hash:
            result['error'] = 'missing_hash'
            return result
        
        # 检查必要的file_start_line和file_end_line
        file_start_line = commit.get('file_start_line')
        file_end_line = commit.get('file_end_line')
        
        if file_start_line is None or file_end_line is None:
            result['error'] = 'missing_file_line_fields'
            return result
        
        # 构建文件路径
        commit_dir = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash)
        
        if not os.path.exists(commit_dir):
            result['error'] = 'commit_dir_not_found'
            return result
        
        # 查找文件
        before_file = find_unique_file_with_pattern(commit_dir, "before.*")
        before_func_file = find_unique_file_with_pattern(commit_dir, "before_func.*")
        
        if not before_file:
            result['error'] = 'before_file_not_found'
            return result
        
        if not before_func_file:
            result['error'] = 'before_func_file_not_found'
            return result
        
        # 计算行号偏移
        line_offset, status = calculate_line_offset(before_file, before_func_file)
        
        if line_offset is None:
            result['error'] = f'offset_calculation_failed: {status}'
            result['status'] = 'partial'
            return result
        
        # 计算func_start_line和func_end_line
        func_start_line = file_start_line - line_offset
        func_end_line = file_end_line - line_offset
        
        # 验证结果合理性
        if func_start_line <= 0 or func_end_line <= 0:
            result['error'] = f'invalid_func_lines: start={func_start_line}, end={func_end_line}'
            result['status'] = 'partial'
            return result
        
        # 计算before_func文件总行数
        before_func_total_lines = count_file_lines(before_func_file)
        
        if before_func_total_lines == 0:
            result['error'] = 'empty_before_func_file'
            result['status'] = 'partial'
            return result
        
        # 验证func_end_line不超过文件总行数
        if func_end_line > before_func_total_lines:
            result['error'] = f'func_end_line ({func_end_line}) exceeds total_lines ({before_func_total_lines})'
            result['status'] = 'partial'
            return result
        
        result.update({
            'line_offset': line_offset,
            'func_start_line': func_start_line,
            'func_end_line': func_end_line,
            'before_func_total_lines': before_func_total_lines,
            'status': 'success'
        })
        
        return result
        
    except Exception as e:
        result['error'] = f'exception: {str(e)}'
        return result

def process_repository_commits(repo):
    """处理单个仓库的所有commits（支持commit级别并行）"""
    repo_name = repo.get('name_long', repo.get('name', ''))
    
    result = {
        'repository_name': repo_name,
        'status': 'failed',
        'commits_total': 0,
        'commits_success': 0,
        'commits_partial': 0,
        'commits_skipped': 0,
        'commits_failed': 0,
        'error': None
    }
    
    try:
        json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, SUMMARY_JSON_FILENAME)
        
        if not os.path.exists(json_file_path):
            result['error'] = 'json_file_not_found'
            return result
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
        
        if not commits:
            result['status'] = 'success'
            result['error'] = 'no_commits'
            return result
        
        result['commits_total'] = len(commits)
        
        # 如果启用commit级别并行处理
        if ENABLE_COMMIT_PARALLEL_PROCESSING and len(commits) > 1:
            # 创建索引映射
            commit_results = [None] * len(commits)
            
            with ThreadPoolExecutor(max_workers=MAX_COMMIT_WORKER_THREADS) as executor:
                future_to_index = {}
                
                for i, commit in enumerate(commits):
                    future = executor.submit(process_single_commit, commit, repo_name)
                    future_to_index[future] = i
                
                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        commit_result = future.result()
                        commit_results[i] = commit_result
                    except Exception as e:
                        commit_results[i] = {
                            'status': 'failed',
                            'error': f'exception: {str(e)}'
                        }
            
            # 应用结果
            for i, commit_result in enumerate(commit_results):
                if commit_result['status'] == 'success':
                    commits[i]['line_offset'] = commit_result['line_offset']
                    commits[i]['func_start_line'] = commit_result['func_start_line']
                    commits[i]['func_end_line'] = commit_result['func_end_line']
                    commits[i]['before_func_total_lines'] = commit_result['before_func_total_lines']
                    result['commits_success'] += 1
                elif commit_result['status'] == 'skipped':
                    result['commits_skipped'] += 1
                elif commit_result['status'] == 'partial':
                    result['commits_partial'] += 1
                else:
                    result['commits_failed'] += 1
        else:
            # 串行处理
            for commit in commits:
                commit_result = process_single_commit(commit, repo_name)
                
                if commit_result['status'] == 'success':
                    commit['line_offset'] = commit_result['line_offset']
                    commit['func_start_line'] = commit_result['func_start_line']
                    commit['func_end_line'] = commit_result['func_end_line']
                    commit['before_func_total_lines'] = commit_result['before_func_total_lines']
                    result['commits_success'] += 1
                elif commit_result['status'] == 'skipped':
                    result['commits_skipped'] += 1
                elif commit_result['status'] == 'partial':
                    result['commits_partial'] += 1
                else:
                    result['commits_failed'] += 1
        
        # 保存结果
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(commits, f, ensure_ascii=False, indent=2)
        
        result['status'] = 'success'
        
    except Exception as e:
        result['error'] = f'repository_processing_error: {str(e)}'
    
    return result

def main():
    """主函数"""
    print("=" * 80)
    print("计算体系结构相关策略的行号偏移")
    print("=" * 80)
    print(f"知识库路径: {KNOWLEDGE_BASE_PATH}")
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"目标JSON文件: {SUMMARY_JSON_FILENAME}")
    print(f"重新处理已有字段: {REGENERATE_EXISTING}")
    print(f"仓库级别并行: {ENABLE_PARALLEL_PROCESSING}")
    print(f"Commit级别并行: {ENABLE_COMMIT_PARALLEL_PROCESSING}")
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
                executor.submit(process_repository_commits, repo): repo 
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
                                global_stats['commits_partial'] += result['commits_partial']
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
            result = process_repository_commits(repo)
            
            global_stats['repositories_total'] += 1
            if result['status'] == 'success':
                global_stats['repositories_success'] += 1
                global_stats['commits_total'] += result['commits_total']
                global_stats['commits_success'] += result['commits_success']
                global_stats['commits_partial'] += result['commits_partial']
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
    print(f"  部分成功: {global_stats['commits_partial']}")
    print(f"  跳过: {global_stats['commits_skipped']}")
    print(f"  失败: {global_stats['commits_failed']}")
    
    if global_stats['commits_total'] > 0:
        success_rate = (global_stats['commits_success'] / global_stats['commits_total']) * 100
        print(f"Commit成功率: {success_rate:.1f}%")

if __name__ == "__main__":
    main()

