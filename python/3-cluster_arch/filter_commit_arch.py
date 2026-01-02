import os
import json
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
INPUT_FILE_NAME = "summary_arch.json"
OUTPUT_FILE_NAME = "summary_filter_arch.json"
MAX_WORKERS = 256
VERBOSE_OUTPUT = False  # 是否输出详细的调试信息
CHECK_FILE_READABLE = True  # 是否检查文件可读性
OVERWRITE_EXISTING = True  # 是否覆盖已存在的输出文件

# 全局统计
error_statistics = defaultdict(int)
error_stats_lock = threading.Lock()

def update_error_stats(error_type):
    """线程安全地更新错误统计"""
    with error_stats_lock:
        error_statistics[error_type] += 1

def log_verbose(message):
    """输出详细调试信息"""
    if VERBOSE_OUTPUT:
        print(f"[详细] {message}")

def check_file_readable(file_path):
    """检查文件是否可读"""
    if not CHECK_FILE_READABLE:
        return True
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.read(1)
        return True
    except Exception:
        return False

def validate_commit(commit, repo_name):
    """验证单个commit是否符合要求"""
    # 1. 基础信息检查
    if not commit.get('hash'):
        return False, "missing_hash"
    
    # 2. 必要字段检查
    if 'func_start_line' not in commit or 'func_end_line' not in commit:
        return False, "missing_line_fields"
    
    if (not isinstance(commit.get('func_start_line'), int) or 
        not isinstance(commit.get('func_end_line'), int)):
        return False, "invalid_line_fields"
    
    # 3. 策略总结字段检查
    if 'optimization_summary_arch_final' not in commit:
        return False, "missing_summary_arch"
    
    if not commit.get('optimization_summary_arch_final'):
        return False, "empty_summary_arch"
    
    # 4. 文件存在性和唯一性检查
    commit_dir = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit['hash'])
    
    if not os.path.exists(commit_dir):
        return False, "commit_dir_not_found"
    
    required_patterns = ['before.*', 'after.*', 'before_func.*', 'after_func.*']
    for pattern in required_patterns:
        files = glob.glob(os.path.join(commit_dir, pattern))
        if len(files) != 1:
            return False, f"file_pattern_{pattern.replace('.*', '')}_count_{len(files)}"
        
        # 5. 文件可读性检查（如果启用）
        if CHECK_FILE_READABLE and not check_file_readable(files[0]):
            return False, f"file_unreadable_{pattern.replace('.*', '')}"
    
    # 6. 检查diff.txt文件
    diff_file = os.path.join(commit_dir, 'diff.txt')
    if not os.path.exists(diff_file):
        return False, "diff_file_not_found"
    
    # 检查diff.txt文件可读性（如果启用）
    if CHECK_FILE_READABLE and not check_file_readable(diff_file):
        return False, "diff_file_unreadable"
    
    return True, "valid"

def process_repository(repo):
    """处理单个代码库"""
    repo_name = repo.get('name_long', repo.get('name', ''))
    json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, INPUT_FILE_NAME)
    filter_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, OUTPUT_FILE_NAME)
    
    # 检查是否覆盖已有文件
    if not OVERWRITE_EXISTING and os.path.exists(filter_file_path):
        return {
            "repo_name": repo_name,
            "status": "skipped_existing",
            "stats": {"total": 0, "valid": 0, "invalid": 0}
        }
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        return {
            "repo_name": repo_name,
            "status": "file_not_found",
            "stats": {"total": 0, "valid": 0, "invalid": 0}
        }
    
    try:
        # 读取commit数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
    except Exception as e:
        return {
            "repo_name": repo_name,
            "status": "read_error",
            "error": str(e),
            "stats": {"total": 0, "valid": 0, "invalid": 0}
        }
    
    if not commits:
        return {
            "repo_name": repo_name,
            "status": "empty_file",
            "stats": {"total": 0, "valid": 0, "invalid": 0}
        }
    
    # 统计变量
    stats = {
        "total": len(commits),
        "valid": 0,
        "invalid": 0
    }
    
    # 验证所有commits
    valid_commits = []
    for commit in commits:
        is_valid, reason = validate_commit(commit, repo_name)
        
        if is_valid:
            valid_commits.append(commit)
            stats["valid"] += 1
        else:
            stats["invalid"] += 1
            update_error_stats(reason)
            if VERBOSE_OUTPUT:
                log_verbose(f"{repo_name} - {commit.get('hash', '未知')[:8]}: {reason}")
    
    # 写入过滤后的结果
    if valid_commits:
        try:
            with open(filter_file_path, 'w', encoding='utf-8') as f:
                json.dump(valid_commits, f, ensure_ascii=False, indent=4)
        except Exception as e:
            return {
                "repo_name": repo_name,
                "status": "write_error",
                "error": str(e),
                "stats": stats
            }
    
    return {
        "repo_name": repo_name,
        "status": "success",
        "stats": stats
    }

def print_error_statistics():
    """打印错误统计信息"""
    if not error_statistics:
        print("\n=== 错误统计 ===")
        print("没有记录到任何错误")
        return
    
    print(f"\n=== 错误统计 ===")
    for error_type, count in sorted(error_statistics.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {error_type.replace('_', ' ')}: {count} 个commit")
    
    total_errors = sum(error_statistics.values())
    print(f"\n错误总计: {total_errors} 个commit")

def main():
    """主函数"""
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"错误：知识库根目录 {KNOWLEDGE_BASE_PATH} 不存在")
        return
    
    print(f"筛选知识库: {KNOWLEDGE_BASE_PATH}")
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"输入文件: {INPUT_FILE_NAME}")
    print(f"输出文件: {OUTPUT_FILE_NAME}")
    print(f"并行线程数: {MAX_WORKERS}")
    print(f"详细输出模式: {'开启' if VERBOSE_OUTPUT else '关闭'}")
    print(f"检查文件可读性: {'开启' if CHECK_FILE_READABLE else '关闭'}")
    print(f"覆盖已有文件: {'是' if OVERWRITE_EXISTING else '否'}")
    
    # 读取代码库列表
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件不存在 - {REPO_LIST_FILE}")
        return
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repositories = json.load(f)
    
    if not repositories:
        print("未找到任何代码库")
        return
    
    print(f"找到 {len(repositories)} 个代码库")
    
    # 总体统计
    total_stats = {
        "repositories": len(repositories),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "total_commits": 0,
        "valid_commits": 0,
        "invalid_commits": 0
    }
    
    # 并行处理所有代码库
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 创建进度条
        pbar = tqdm(total=len(repositories), desc="处理代码库", unit="repo")
        
        # 提交任务
        future_to_repo = {executor.submit(process_repository, repo): repo for repo in repositories}
        
        results = []
        for future in as_completed(future_to_repo):
            repo = future_to_repo[future]
            repo_name = repo.get('name_long', repo.get('name', ''))
            try:
                result = future.result()
                results.append(result)
                
                # 更新统计
                if result["status"] == "success":
                    total_stats["success"] += 1
                    total_stats["total_commits"] += result["stats"]["total"]
                    total_stats["valid_commits"] += result["stats"]["valid"]
                    total_stats["invalid_commits"] += result["stats"]["invalid"]
                elif result["status"] == "skipped_existing":
                    total_stats["skipped"] += 1
                else:
                    total_stats["failed"] += 1
                
                pbar.set_postfix({
                    "成功": total_stats["success"],
                    "跳过": total_stats["skipped"],
                    "有效commit": total_stats["valid_commits"]
                })
                
            except Exception as e:
                print(f"\n[异常] 代码库 {repo_name}: {str(e)}")
                total_stats["failed"] += 1
            
            pbar.update(1)
        
        pbar.close()
    
    # 打印结果
    print(f"\n=== 处理结果 ===")
    for result in results:
        if result["status"] == "success":
            stats = result["stats"]
            print(f"{result['repo_name']}: 总计{stats['total']}, 有效{stats['valid']}, 无效{stats['invalid']}")
        elif result["status"] == "skipped_existing":
            print(f"{result['repo_name']}: 跳过（文件已存在）")
        else:
            print(f"{result['repo_name']}: {result['status']} - {result.get('error', '')}")
    
    # 打印总体统计
    print(f"\n=== 总体统计 ===")
    print(f"代码库总数: {total_stats['repositories']}")
    print(f"成功处理: {total_stats['success']}")
    print(f"跳过处理: {total_stats['skipped']}")
    print(f"处理失败: {total_stats['failed']}")
    print(f"总commit数: {total_stats['total_commits']}")
    print(f"有效commit数: {total_stats['valid_commits']}")
    print(f"无效commit数: {total_stats['invalid_commits']}")
    
    # 计算有效率
    if total_stats['total_commits'] > 0:
        valid_rate = (total_stats['valid_commits'] / total_stats['total_commits']) * 100
        print(f"有效率: {valid_rate:.1f}%")
    
    # 打印错误统计
    print_error_statistics()
    
    print(f"\n处理完成！")

if __name__ == "__main__":
    main()

