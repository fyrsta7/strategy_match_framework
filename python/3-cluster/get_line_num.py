import os
import json
import glob
import difflib
import re
import sys
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ==================== 全局配置 ====================
# 知识库根路径
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base_all")

# 目标JSON文件名
SUMMARY_JSON_FILENAME = "summary.json"

# 是否重新处理已有字段
REGENERATE_EXISTING = True

# 并行处理配置
MAX_WORKER_THREADS = 32
ENABLE_PARALLEL_PROCESSING = True
RETRY_COUNT = 3
RETRY_DELAY = 1.0

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

# ==================== 工具函数 ====================
def find_file_with_pattern(directory: str, pattern: str) -> Optional[str]:
    """
    查找符合精确模式的文件
    pattern: 'before' -> 匹配 'before.*'
    pattern: 'after' -> 匹配 'after.*'  
    pattern: 'before_func' -> 匹配 'before_func.*'
    
    只有找到唯一一个符合要求的文件时才返回，否则抛出异常
    """
    if not os.path.exists(directory):
        return None
    
    # 尝试精确匹配（无扩展名的文件）
    exact_pattern = os.path.join(directory, pattern)
    if os.path.exists(exact_pattern):
        return exact_pattern
    
    # 使用精确的点分隔模式：pattern.*
    # 这确保 'before' 只匹配 'before.xxx'，不匹配 'before_func.xxx'
    pattern_files = glob.glob(os.path.join(directory, pattern + '.*'))
    
    if len(pattern_files) == 0:
        raise FileNotFoundError(f"No file found matching pattern '{pattern}.*' in {directory}")
    elif len(pattern_files) > 1:
        raise ValueError(f"Multiple files found matching pattern '{pattern}.*' in {directory}: {pattern_files}")
    else:
        # 正好找到一个文件
        return pattern_files[0]

def read_file_with_encoding(file_path: str) -> List[str]:
    """使用多种编码尝试读取文件"""
    if not file_path or not os.path.exists(file_path):
        return []
    
    encodings = ['utf-8', 'gbk', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            # 标准化行尾符
            return [line.rstrip('\r\n') + '\n' for line in lines]
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading {file_path} with {encoding}: {e}")
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
    
    # 移除最后的空行（如果存在）
    while lines and lines[-1].strip() == '':
        lines.pop()
    
    # 确保最后一行有换行符
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    
    return lines

def calculate_file_diff(before_lines: List[str], after_lines: List[str]) -> Tuple[Optional[int], Optional[int], float]:
    """
    使用difflib计算文件diff，返回修改的起始和结束行号
    返回: (start_line, end_line, confidence)
    """
    if not before_lines or not after_lines:
        return None, None, 0.0
    
    try:
        # 使用unified_diff
        diff_lines = list(difflib.unified_diff(
            before_lines, 
            after_lines, 
            lineterm='',
            n=0  # 不包含上下文行
        ))
        
        if not diff_lines:
            # 文件没有变化
            return None, None, 0.0
        
        # 解析hunk信息
        hunk_info = []
        for line in diff_lines:
            if line.startswith('@@'):
                # 解析 @@ -old_start,old_count +new_start,new_count @@
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
        
        # 计算修改范围（基于before文件的行号）
        start_line = min(hunk['old_start'] for hunk in hunk_info)
        
        # 计算end_line：考虑所有hunk的范围
        max_end = 0
        for hunk in hunk_info:
            if hunk['old_count'] > 0:
                # 有删除行的情况
                end = hunk['old_start'] + hunk['old_count'] - 1
            else:
                # 纯新增的情况
                end = hunk['old_start']
            max_end = max(max_end, end)
        
        end_line = max_end
        
        # 验证结果合理性
        if start_line <= 0 or end_line <= 0 or start_line > len(before_lines):
            return None, None, 0.0
        
        # 确保end_line不超过文件长度
        end_line = min(end_line, len(before_lines))
        
        # 确保start_line <= end_line
        if start_line > end_line:
            end_line = start_line
        
        # 计算置信度（基于hunk数量和复杂度）
        confidence = min(1.0, 0.8 + 0.1 * min(len(hunk_info), 2))
        
        return start_line, end_line, confidence
        
    except Exception as e:
        print(f"Error in calculate_file_diff: {e}")
        return None, None, 0.0

def fallback_diff_calculation(before_lines: List[str], after_lines: List[str]) -> Tuple[Optional[int], Optional[int], float]:
    """
    备选diff计算方法，使用SequenceMatcher
    """
    try:
        matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
        opcodes = matcher.get_opcodes()
        
        # 找到所有的修改操作
        modified_ranges = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag != 'equal':  # 'delete', 'insert', 'replace'
                if tag == 'insert' and i1 == i2:
                    # 纯插入，使用插入位置
                    modified_ranges.append((i1 + 1, i1 + 1))
                else:
                    # 删除或替换，使用原文件的行号
                    modified_ranges.append((i1 + 1, i2))
        
        if not modified_ranges:
            return None, None, 0.0
        
        start_line = min(start for start, _ in modified_ranges)
        end_line = max(end for _, end in modified_ranges)
        
        return start_line, end_line, 0.6  # 较低的置信度
        
    except Exception as e:
        print(f"Error in fallback_diff_calculation: {e}")
        return None, None, 0.0

def process_commit_diff(commit: dict) -> dict:
    """
    处理单个commit的diff计算
    返回包含结果和状态信息的字典
    """
    result = {
        'file_start_line': None,
        'file_end_line': None,
        'confidence': 0.0,
        'status': 'failed',
        'error': None
    }
    
    try:
        # 检查是否需要重新处理
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
        
        # 获取必要信息
        repo_name = commit.get('repository_name')
        commit_hash = commit.get('hash')
        
        if not repo_name or not commit_hash:
            result['error'] = 'Missing repository_name or hash'
            return result
        
        # 构建文件路径
        commit_dir = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash)
        
        if not os.path.exists(commit_dir):
            result['error'] = f'Commit directory not found: {commit_dir}'
            return result
        
        # 查找before和after文件
        before_file = find_file_with_pattern(commit_dir, "before")
        after_file = find_file_with_pattern(commit_dir, "after")
        
        if not before_file:
            result['error'] = 'Before file not found'
            return result
        
        if not after_file:
            result['error'] = 'After file not found'
            return result
        
        # 读取文件内容
        before_lines = read_file_with_encoding(before_file)
        after_lines = read_file_with_encoding(after_file)
        
        if not before_lines:
            result['error'] = 'Before file is empty or unreadable'
            return result
        
        if not after_lines:
            result['error'] = 'After file is empty or unreadable'
            return result
        
        # 标准化文件内容
        before_lines = normalize_file_content(before_lines)
        after_lines = normalize_file_content(after_lines)
        
        # 计算diff
        start_line, end_line, confidence = calculate_file_diff(before_lines, after_lines)
        
        if start_line is None or end_line is None:
            # 尝试备选方法
            start_line, end_line, confidence = fallback_diff_calculation(before_lines, after_lines)
        
        if start_line is None or end_line is None:
            result['error'] = 'Failed to calculate diff with both methods'
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

# ==================== 新增：代码库发现和处理函数 ====================
def discover_repositories() -> List[Dict[str, Any]]:
    """
    发现知识库中的所有代码库
    返回代码库信息列表
    """
    repositories = []
    
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"错误: 知识库目录不存在 - {KNOWLEDGE_BASE_PATH}")
        return repositories
    
    # 遍历知识库目录下的所有子目录
    for item in os.listdir(KNOWLEDGE_BASE_PATH):
        repo_path = os.path.join(KNOWLEDGE_BASE_PATH, item)
        
        # 跳过非目录项
        if not os.path.isdir(repo_path):
            continue
        
        # 检查summary.json文件是否存在
        summary_json_path = os.path.join(repo_path, SUMMARY_JSON_FILENAME)
        
        repo_info = {
            'name': item,
            'path': repo_path,
            'summary_json_path': summary_json_path,
            'has_summary_json': os.path.exists(summary_json_path),
            'error': None
        }
        
        if not repo_info['has_summary_json']:
            repo_info['error'] = f"Summary JSON file not found: {summary_json_path}"
        
        repositories.append(repo_info)
    
    return repositories

def load_repository_commits(repo_info: Dict[str, Any]) -> Tuple[List[dict], Optional[str]]:
    """
    加载代码库的commit信息
    返回 (commits_list, error_message)
    """
    if not repo_info['has_summary_json']:
        return [], repo_info['error']
    
    try:
        with open(repo_info['summary_json_path'], 'r', encoding='utf-8') as f:
            commits = json.load(f)
        
        if not isinstance(commits, list):
            return [], "Summary JSON is not a list"
        
        return commits, None
        
    except json.JSONDecodeError as e:
        return [], f"JSON decode error: {str(e)}"
    except Exception as e:
        return [], f"Error loading commits: {str(e)}"

def save_repository_commits(repo_info: Dict[str, Any], commits: List[dict]) -> Optional[str]:
    """
    保存代码库的commit信息
    返回错误信息（如果有）
    """
    try:
        # 使用文件锁确保原子性写入
        temp_path = repo_info['summary_json_path'] + '.tmp'
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(commits, f, ensure_ascii=False, indent=2)
        
        # 原子性替换文件
        if os.path.exists(repo_info['summary_json_path']):
            os.replace(temp_path, repo_info['summary_json_path'])
        else:
            os.rename(temp_path, repo_info['summary_json_path'])
        
        return None
        
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return f"Error saving commits: {str(e)}"

def process_single_repository(repo_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个代码库
    返回处理结果统计
    """
    result = {
        'repository_name': repo_info['name'],
        'status': 'failed',
        'commits_total': 0,
        'commits_success': 0,
        'commits_skipped': 0,
        'commits_failed': 0,
        'error': None,
        'processing_time': 0.0
    }
    
    start_time = time.time()
    
    try:
        print(f"开始处理代码库: {repo_info['name']}")
        
        # 加载commits
        commits, error = load_repository_commits(repo_info)
        if error:
            result['error'] = error
            return result
        
        if not commits:
            result['status'] = 'success'
            result['error'] = 'No commits found'
            return result
        
        result['commits_total'] = len(commits)
        
        # 处理每个commit
        for i, commit in enumerate(commits):
            commit_hash = commit.get('hash', 'unknown')
            
            print(f"  处理 [{i+1}/{len(commits)}] {commit_hash[:8]}...", end=' ')
            
            # 执行diff计算（带重试机制）
            commit_result = None
            for attempt in range(RETRY_COUNT):
                try:
                    commit_result = process_commit_diff(commit)
                    break
                except Exception as e:
                    if attempt == RETRY_COUNT - 1:
                        commit_result = {
                            'status': 'failed',
                            'error': f'Max retries exceeded: {str(e)}'
                        }
                    else:
                        time.sleep(RETRY_DELAY)
                        continue
            
            # 更新commit信息
            if commit_result['status'] == 'success':
                commit['file_start_line'] = commit_result['file_start_line']
                commit['file_end_line'] = commit_result['file_end_line']
                result['commits_success'] += 1
                print(f"✓ ({commit_result['file_start_line']}-{commit_result['file_end_line']}, conf={commit_result['confidence']:.2f})")
            elif commit_result['status'] == 'skipped':
                result['commits_skipped'] += 1
                print("⏭ (已存在)")
            else:
                result['commits_failed'] += 1
                print(f"✗ ({commit_result['error']})")
        
        # 保存结果
        save_error = save_repository_commits(repo_info, commits)
        if save_error:
            result['error'] = save_error
            return result
        
        result['status'] = 'success'
        
    except Exception as e:
        result['error'] = f'Repository processing error: {str(e)}'
        print(f"代码库处理异常: {repo_info['name']} - {str(e)}")
        traceback.print_exc()
    
    finally:
        result['processing_time'] = time.time() - start_time
    
    return result

def update_global_stats(repo_result: Dict[str, Any]):
    """更新全局统计信息"""
    with stats_lock:
        global_stats['repositories_total'] += 1
        if repo_result['status'] == 'success':
            global_stats['repositories_success'] += 1
        else:
            global_stats['repositories_failed'] += 1
        
        global_stats['commits_total'] += repo_result['commits_total']
        global_stats['commits_success'] += repo_result['commits_success']
        global_stats['commits_skipped'] += repo_result['commits_skipped']
        global_stats['commits_failed'] += repo_result['commits_failed']

def main():
    """主函数"""
    print("=" * 80)
    print("自动处理所有代码库的 commit diff 计算")
    print("=" * 80)
    print(f"知识库路径: {KNOWLEDGE_BASE_PATH}")
    print(f"目标JSON文件: {SUMMARY_JSON_FILENAME}")
    print(f"重新处理已有字段: {REGENERATE_EXISTING}")
    print(f"并行处理: {ENABLE_PARALLEL_PROCESSING}")
    if ENABLE_PARALLEL_PROCESSING:
        print(f"最大工作线程数: {MAX_WORKER_THREADS}")
    print("-" * 80)
    
    # 发现所有代码库
    print("正在发现代码库...")
    repositories = discover_repositories()
    
    if not repositories:
        print("未发现任何代码库")
        return
    
    # 过滤出有效的代码库
    valid_repositories = [repo for repo in repositories if repo['has_summary_json']]
    invalid_repositories = [repo for repo in repositories if not repo['has_summary_json']]
    
    print(f"发现代码库总数: {len(repositories)}")
    print(f"有效代码库: {len(valid_repositories)}")
    print(f"无效代码库: {len(invalid_repositories)}")
    
    if invalid_repositories:
        print("\n无效代码库列表:")
        for repo in invalid_repositories:
            print(f"  - {repo['name']}: {repo['error']}")
    
    if not valid_repositories:
        print("没有有效的代码库需要处理")
        return
    
    print(f"\n开始处理 {len(valid_repositories)} 个代码库...")
    print("-" * 80)
    
    # 处理代码库
    start_time = time.time()
    
    if ENABLE_PARALLEL_PROCESSING and len(valid_repositories) > 1:
        # 并行处理
        with ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
            # 提交所有任务
            future_to_repo = {
                executor.submit(process_single_repository, repo): repo 
                for repo in valid_repositories
            }
            
            # 收集结果
            for future in as_completed(future_to_repo):
                repo_info = future_to_repo[future]
                try:
                    result = future.result()
                    update_global_stats(result)
                    
                    if result['status'] == 'success':
                        print(f"✓ 完成: {result['repository_name']} ({result['processing_time']:.1f}s)")
                    else:
                        print(f"✗ 失败: {result['repository_name']} - {result['error']}")
                        
                except Exception as e:
                    print(f"✗ 异常: {repo_info['name']} - {str(e)}")
                    with stats_lock:
                        global_stats['repositories_total'] += 1
                        global_stats['repositories_failed'] += 1
    else:
        # 串行处理
        for repo in valid_repositories:
            result = process_single_repository(repo)
            update_global_stats(result)
            
            if result['status'] == 'success':
                print(f"✓ 完成: {result['repository_name']} ({result['processing_time']:.1f}s)")
            else:
                print(f"✗ 失败: {result['repository_name']} - {result['error']}")
    
    # 输出最终统计
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("处理完成 - 最终统计")
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
    
    if global_stats['repositories_total'] > 0:
        repo_success_rate = (global_stats['repositories_success'] / global_stats['repositories_total']) * 100
        print(f"代码库成功率: {repo_success_rate:.1f}%")
    
    if (global_stats['commits_total'] - global_stats['commits_skipped']) > 0:
        commit_success_rate = (global_stats['commits_success'] / (global_stats['commits_total'] - global_stats['commits_skipped'])) * 100
        print(f"Commit成功率: {commit_success_rate:.1f}%")

if __name__ == "__main__":
    main()