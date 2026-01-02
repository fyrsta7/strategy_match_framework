import os
import json
import glob
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
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")

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
    'commits_partial': 0,
    'commits_skipped': 0,
    'commits_failed': 0
}

# ==================== 工具函数 ====================
def find_unique_file_with_pattern(directory: str, pattern: str) -> Optional[str]:
    """查找符合模式的文件，要求必须唯一匹配"""
    if not os.path.exists(directory):
        return None
    
    # 使用通配符匹配
    pattern_files = glob.glob(os.path.join(directory, pattern))
    
    if len(pattern_files) == 0:
        return None
    elif len(pattern_files) == 1:
        return pattern_files[0]
    else:
        # 多个匹配，这是错误情况
        print(f"错误: 模式 '{pattern}' 在目录 '{directory}' 中匹配到多个文件: {pattern_files}")
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
            # 保持原始行内容（包括换行符）
            return lines
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading {file_path} with {encoding}: {e}")
            continue
    
    # 最后尝试忽略错误
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
    if not line:
        return ""
    
    # 移除行尾换行符
    line = line.rstrip('\r\n')
    
    # 移除前后空白
    line = line.strip()
    
    # 移除多余的空白字符
    import re
    line = re.sub(r'\s+', ' ', line)
    
    return line

def extract_feature_lines(lines: List[str], max_lines: int = 5) -> List[Tuple[int, str]]:
    """
    从文件中提取特征行用于匹配
    返回: [(行号, 标准化内容), ...]
    """
    feature_lines = []
    
    for i, line in enumerate(lines):
        normalized = normalize_line_for_matching(line)
        
        # 跳过空行和简单行
        if not normalized or len(normalized) < 5:
            continue
        
        # 跳过注释行（简单判断）
        if normalized.startswith('//') or normalized.startswith('#') or normalized.startswith('/*'):
            continue
        
        feature_lines.append((i + 1, normalized))  # 行号从1开始
        
        if len(feature_lines) >= max_lines:
            break
    
    return feature_lines

def calculate_line_similarity(line1: str, line2: str) -> float:
    """计算两行的相似度"""
    if line1 == line2:
        return 1.0
    
    if not line1 or not line2:
        return 0.0
    
    # 简单的字符匹配率
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, line1, line2)
    return matcher.ratio()

def find_best_match_position(feature_lines: List[Tuple[int, str]], 
                           target_lines: List[str], 
                           min_confidence: float = 0.8) -> Tuple[Optional[int], float]:
    """
    在目标文件中寻找特征行序列的最佳匹配位置
    返回: (匹配的起始行号, 置信度)
    """
    if not feature_lines or not target_lines:
        return None, 0.0
    
    best_position = None
    best_confidence = 0.0
    
    # 获取第一个特征行作为主要搜索目标
    first_feature_line_num, first_feature_content = feature_lines[0]
    
    # 在目标文件中搜索
    for target_line_num, target_line in enumerate(target_lines, 1):
        target_normalized = normalize_line_for_matching(target_line)
        
        # 计算第一行的匹配度
        similarity = calculate_line_similarity(first_feature_content, target_normalized)
        
        if similarity < 0.7:  # 第一行匹配度太低，跳过
            continue
        
        # 如果只有一个特征行，直接返回
        if len(feature_lines) == 1:
            if similarity >= min_confidence:
                return target_line_num, similarity
            continue
        
        # 验证后续特征行
        total_similarity = similarity
        matched_lines = 1
        
        for j, (relative_line_num, feature_content) in enumerate(feature_lines[1:], 1):
            # 计算相对位置
            expected_target_line_num = target_line_num + (relative_line_num - first_feature_line_num)
            
            if expected_target_line_num <= len(target_lines):
                expected_target_line = target_lines[expected_target_line_num - 1]
                expected_target_normalized = normalize_line_for_matching(expected_target_line)
                
                line_similarity = calculate_line_similarity(feature_content, expected_target_normalized)
                total_similarity += line_similarity
                matched_lines += 1
            else:
                # 超出文件范围，降低置信度
                total_similarity += 0.5
                matched_lines += 1
        
        # 计算平均置信度
        avg_confidence = total_similarity / matched_lines
        
        if avg_confidence > best_confidence and avg_confidence >= min_confidence:
            best_confidence = avg_confidence
            best_position = target_line_num
    
    return best_position, best_confidence

def calculate_line_offset(before_func_file: str, before_file: str) -> Tuple[Optional[int], float]:
    """
    计算line_offset
    返回: (line_offset, confidence)
    """
    try:
        # 读取文件
        before_func_lines = read_file_with_encoding(before_func_file)
        before_lines = read_file_with_encoding(before_file)
        
        if not before_func_lines or not before_lines:
            return None, 0.0
        
        # 提取before_func的特征行
        feature_lines = extract_feature_lines(before_func_lines, max_lines=3)
        
        if not feature_lines:
            return None, 0.0
        
        # 在before文件中寻找匹配位置
        match_position, confidence = find_best_match_position(feature_lines, before_lines)
        
        if match_position is None:
            return None, 0.0
        
        # 计算line_offset
        # before_func文件的第一个特征行对应before文件中的match_position
        func_first_line = feature_lines[0][0]  # before_func中的行号
        line_offset = match_position - func_first_line
        
        return line_offset, confidence
        
    except Exception as e:
        print(f"Error calculating line_offset: {e}")
        return None, 0.0

def process_commit_func_info(commit: dict) -> dict:
    """
    处理单个commit的函数信息计算
    """
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
            required_fields = ['line_offset', 'func_start_line', 'func_end_line', 'before_func_total_lines']
            if all(field in commit and commit[field] is not None for field in required_fields):
                result.update({
                    'line_offset': commit['line_offset'],
                    'func_start_line': commit['func_start_line'],
                    'func_end_line': commit['func_end_line'],
                    'before_func_total_lines': commit['before_func_total_lines'],
                    'status': 'skipped'
                })
                return result
        
        # 获取必要信息
        repo_name = commit.get('repository_name')
        commit_hash = commit.get('hash')
        
        if not repo_name or not commit_hash:
            result['error'] = 'Missing repository_name or hash'
            return result
        
        # 检查是否有file_start_line和file_end_line
        file_start_line = commit.get('file_start_line')
        file_end_line = commit.get('file_end_line')
        
        if file_start_line is None or file_end_line is None:
            result['error'] = 'Missing file_start_line or file_end_line'
            return result
        
        # 构建文件路径
        commit_dir = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash)
        
        if not os.path.exists(commit_dir):
            result['error'] = f'Commit directory not found: {commit_dir}'
            return result
        
        # 查找必要文件
        before_file = find_unique_file_with_pattern(commit_dir, "before.*")
        before_func_file = find_unique_file_with_pattern(commit_dir, "before_func.*")
        
        if not before_file:
            result['error'] = 'Before file not found'
            return result
        
        if not before_func_file:
            result['error'] = 'Before_func file not found'
            return result
        
        # 计算before_func_total_lines
        total_lines = count_file_lines(before_func_file)
        result['before_func_total_lines'] = total_lines
        
        # 计算line_offset
        line_offset, confidence = calculate_line_offset(before_func_file, before_file)
        
        if line_offset is None:
            result['error'] = 'Failed to calculate line_offset'
            # 即使line_offset失败，也要保存before_func_total_lines
            result['status'] = 'partial'
            return result
        
        result['line_offset'] = line_offset
        
        # 计算func_start_line和func_end_line
        func_start_line = file_start_line - line_offset
        func_end_line = file_end_line - line_offset
        
        # 验证结果合理性
        if func_start_line <= 0 or func_end_line <= 0 or func_start_line > total_lines:
            result['error'] = f'Invalid func line range: {func_start_line}-{func_end_line} (total: {total_lines})'
            result['status'] = 'partial'
            return result
        
        # 确保func_end_line不超过文件长度
        func_end_line = min(func_end_line, total_lines)
        
        # 确保func_start_line <= func_end_line
        if func_start_line > func_end_line:
            func_end_line = func_start_line
        
        result.update({
            'func_start_line': func_start_line,
            'func_end_line': func_end_line,
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
        'commits_partial': 0,
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
            
            # 执行函数信息计算（带重试机制）
            commit_result = None
            for attempt in range(RETRY_COUNT):
                try:
                    commit_result = process_commit_func_info(commit)
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
                commit['line_offset'] = commit_result['line_offset']
                commit['func_start_line'] = commit_result['func_start_line']
                commit['func_end_line'] = commit_result['func_end_line']
                commit['before_func_total_lines'] = commit_result['before_func_total_lines']
                result['commits_success'] += 1
                print(f"✓ (offset={commit_result['line_offset']}, func={commit_result['func_start_line']}-{commit_result['func_end_line']}, total={commit_result['before_func_total_lines']})")
            elif commit_result['status'] == 'partial':
                if commit_result['before_func_total_lines'] is not None:
                    commit['before_func_total_lines'] = commit_result['before_func_total_lines']
                if commit_result['line_offset'] is not None:
                    commit['line_offset'] = commit_result['line_offset']
                result['commits_partial'] += 1
                print(f"◐ 部分成功 ({commit_result['error']})")
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
        global_stats['commits_partial'] += repo_result['commits_partial']
        global_stats['commits_skipped'] += repo_result['commits_skipped']
        global_stats['commits_failed'] += repo_result['commits_failed']

def main():
    """主函数"""
    print("=" * 80)
    print("自动处理所有代码库的函数信息计算")
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
    print(f"  完全成功: {global_stats['commits_success']}")
    print(f"  部分成功: {global_stats['commits_partial']}")
    print(f"  跳过: {global_stats['commits_skipped']}")
    print(f"  失败: {global_stats['commits_failed']}")
    
    if global_stats['repositories_total'] > 0:
        repo_success_rate = (global_stats['repositories_success'] / global_stats['repositories_total']) * 100
        print(f"代码库成功率: {repo_success_rate:.1f}%")
    
    if (global_stats['commits_total'] - global_stats['commits_skipped']) > 0:
        processed = global_stats['commits_total'] - global_stats['commits_skipped']
        complete_success_rate = (global_stats['commits_success'] / processed) * 100
        useful_success_rate = ((global_stats['commits_success'] + global_stats['commits_partial']) / processed) * 100
        print(f"Commit完全成功率: {complete_success_rate:.1f}%")
        print(f"Commit有效成功率: {useful_success_rate:.1f}%")

if __name__ == "__main__":
    main()