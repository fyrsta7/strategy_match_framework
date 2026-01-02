#!/usr/bin/env python3
"""
检查 commit 详细信息文件夹是否已正确复制
验证 semopt_commit_list/knowledge_base/ 中每个 commit 的详细信息文件夹是否存在且完整
"""

import json
import os
import sys
import glob
from collections import defaultdict
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count

# 添加 python/ 目录到 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
SOURCE_KNOWLEDGE_BASE = os.path.join(config.root_path, "knowledge_base")
TARGET_KNOWLEDGE_BASE = os.path.join(config.root_path, "semopt_commit_list", "knowledge_base")
COMMIT_LIST_FILE = os.path.join(config.root_path, "all_is_opt_final.json")

# 并行处理配置
NUM_PROCESSES = min(cpu_count(), 128)  # 最多使用128个进程

# 需要检查的文件类型
EXPECTED_FILE_TYPES = ["diff.txt", "before", "after", "before_func", "after_func"]

# 排除的文件模式
EXCLUDE_PATTERNS = [
    "_no_comment",
    "_api.json",
    "rapgen_",
]


def should_exclude_file(filename):
    """判断文件是否应该被排除"""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filename:
            return True
    return False


def find_commit_files(commit_dir):
    """
    在 commit 目录中查找需要的 5 类文件
    返回: {file_type: file_path} 的字典
    """
    found_files = {}
    
    if not os.path.exists(commit_dir):
        return found_files
    
    try:
        all_files = os.listdir(commit_dir)
    except Exception as e:
        return found_files
    
    # 查找 diff.txt
    if "diff.txt" in all_files:
        found_files["diff.txt"] = "diff.txt"
    
    # 查找 before.* (排除 before_func.*)
    for f in all_files:
        if f.startswith("before.") and not f.startswith("before_func.") and not should_exclude_file(f):
            found_files["before"] = f
            break
    
    # 查找 after.* (排除 after_func.*)
    for f in all_files:
        if f.startswith("after.") and not f.startswith("after_func.") and not should_exclude_file(f):
            found_files["after"] = f
            break
    
    # 查找 before_func.*
    for f in all_files:
        if f.startswith("before_func.") and not should_exclude_file(f):
            found_files["before_func"] = f
            break
    
    # 查找 after_func.*
    for f in all_files:
        if f.startswith("after_func.") and not should_exclude_file(f):
            found_files["after_func"] = f
            break
    
    return found_files


def check_commit_directory(args):
    """
    检查单个 commit 的详细信息目录
    
    Args:
        args: (repo_name, commit_hash) 元组
    
    Returns:
        dict: 检查结果
    """
    repo_name, commit_hash = args
    
    result = {
        "repo_name": repo_name,
        "commit_hash": commit_hash,
        "target_exists": False,
        "source_exists": False,
        "files_status": {},
        "missing_files": [],
        "empty_files": [],
        "file_count": 0,
        "status": "unknown",
    }
    
    # 构建路径
    source_commit_dir = os.path.join(
        SOURCE_KNOWLEDGE_BASE,
        repo_name,
        "modified_file",
        commit_hash
    )
    
    target_commit_dir = os.path.join(
        TARGET_KNOWLEDGE_BASE,
        repo_name,
        "modified_file",
        commit_hash
    )
    
    # 检查源目录
    result["source_exists"] = os.path.exists(source_commit_dir) and os.path.isdir(source_commit_dir)
    
    # 检查目标目录
    result["target_exists"] = os.path.exists(target_commit_dir) and os.path.isdir(target_commit_dir)
    
    if not result["target_exists"]:
        result["status"] = "target_missing"
        return result
    
    # 查找目标目录中的文件
    target_files = find_commit_files(target_commit_dir)
    
    # 统计文件数量
    try:
        all_files = os.listdir(target_commit_dir)
        result["file_count"] = len([f for f in all_files if os.path.isfile(os.path.join(target_commit_dir, f))])
    except Exception:
        pass
    
    # 检查每个必需的文件类型
    for file_type in EXPECTED_FILE_TYPES:
        if file_type in target_files:
            filename = target_files[file_type]
            file_path = os.path.join(target_commit_dir, filename)
            
            # 检查文件是否存在且非空
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                result["files_status"][file_type] = {
                    "exists": True,
                    "filename": filename,
                    "size": file_size,
                    "empty": file_size == 0
                }
                if file_size == 0:
                    result["empty_files"].append(file_type)
            else:
                result["files_status"][file_type] = {
                    "exists": False,
                    "filename": filename,
                }
                result["missing_files"].append(file_type)
        else:
            result["files_status"][file_type] = {"exists": False}
            result["missing_files"].append(file_type)
    
    # 判断状态
    if len(result["missing_files"]) == 0 and len(result["empty_files"]) == 0:
        result["status"] = "complete"
    elif len(result["missing_files"]) < len(EXPECTED_FILE_TYPES):
        result["status"] = "partial"
    else:
        result["status"] = "incomplete"
    
    # 如果源目录存在，进行对比检查
    if result["source_exists"]:
        source_files = find_commit_files(source_commit_dir)
        result["source_file_count"] = len(source_files)
        result["target_file_count"] = len(target_files)
        result["files_match"] = set(source_files.keys()) == set(target_files.keys())
    
    return result


def merge_results(results):
    """合并检查结果"""
    merged = {
        "total_commits": len(results),
        "complete": 0,
        "partial": 0,
        "incomplete": 0,
        "target_missing": 0,
        "source_missing": 0,
        "missing_files_stats": defaultdict(int),
        "empty_files_stats": defaultdict(int),
        "file_count_stats": {
            "min": float('inf'),
            "max": 0,
            "total": 0,
            "count": 0,
        },
        "repo_stats": defaultdict(lambda: {
            "total": 0,
            "complete": 0,
            "partial": 0,
            "incomplete": 0,
            "target_missing": 0,
        }),
        "issues": {
            "target_missing": [],
            "missing_files": [],
            "empty_files": [],
            "source_missing_but_target_exists": [],
        },
    }
    
    for result in results:
        status = result["status"]
        repo_name = result["repo_name"]
        
        merged["repo_stats"][repo_name]["total"] += 1
        
        if status == "complete":
            merged["complete"] += 1
            merged["repo_stats"][repo_name]["complete"] += 1
        elif status == "partial":
            merged["partial"] += 1
            merged["repo_stats"][repo_name]["partial"] += 1
        elif status == "incomplete":
            merged["incomplete"] += 1
            merged["repo_stats"][repo_name]["incomplete"] += 1
        elif status == "target_missing":
            merged["target_missing"] += 1
            merged["repo_stats"][repo_name]["target_missing"] += 1
            merged["issues"]["target_missing"].append({
                "repo": repo_name,
                "commit": result["commit_hash"],
            })
        
        # 统计缺失的文件类型
        for file_type in result["missing_files"]:
            merged["missing_files_stats"][file_type] += 1
            if len(merged["issues"]["missing_files"]) < 100:  # 只保存前100个
                merged["issues"]["missing_files"].append({
                    "repo": repo_name,
                    "commit": result["commit_hash"],
                    "file_type": file_type,
                })
        
        # 统计空文件
        for file_type in result["empty_files"]:
            merged["empty_files_stats"][file_type] += 1
            if len(merged["issues"]["empty_files"]) < 100:  # 只保存前100个
                merged["issues"]["empty_files"].append({
                    "repo": repo_name,
                    "commit": result["commit_hash"],
                    "file_type": file_type,
                })
        
        # 统计文件数量
        if result["file_count"] > 0:
            merged["file_count_stats"]["min"] = min(merged["file_count_stats"]["min"], result["file_count"])
            merged["file_count_stats"]["max"] = max(merged["file_count_stats"]["max"], result["file_count"])
            merged["file_count_stats"]["total"] += result["file_count"]
            merged["file_count_stats"]["count"] += 1
        
        # 检查源目录不存在但目标目录存在的情况
        if not result["source_exists"] and result["target_exists"]:
            merged["issues"]["source_missing_but_target_exists"].append({
                "repo": repo_name,
                "commit": result["commit_hash"],
            })
        
        if not result["source_exists"]:
            merged["source_missing"] += 1
    
    # 计算文件数量平均值
    if merged["file_count_stats"]["count"] > 0:
        merged["file_count_stats"]["avg"] = merged["file_count_stats"]["total"] / merged["file_count_stats"]["count"]
    else:
        merged["file_count_stats"]["avg"] = 0
        merged["file_count_stats"]["min"] = 0
    
    return merged


def main():
    """主函数"""
    print("="*80)
    print("检查 Commit 详细信息文件夹")
    print("="*80)
    print(f"源目录: {SOURCE_KNOWLEDGE_BASE}")
    print(f"目标目录: {TARGET_KNOWLEDGE_BASE}")
    print(f"Commit 列表文件: {COMMIT_LIST_FILE}")
    print(f"并行进程数: {NUM_PROCESSES}")
    print("="*80)
    
    start_time = time.time()
    
    # 读取 commit 列表
    print("\n读取 commit 列表...")
    if not os.path.exists(COMMIT_LIST_FILE):
        print(f"错误: 文件不存在: {COMMIT_LIST_FILE}")
        return
    
    with open(COMMIT_LIST_FILE, 'r', encoding='utf-8') as f:
        commits = json.load(f)
    
    print(f"共读取 {len(commits)} 个 commits")
    
    # 提取 (repo_name, commit_hash) 对
    commit_tasks = []
    for commit in commits:
        repo_name = commit.get("repository_name")
        commit_hash = commit.get("hash")
        if repo_name and commit_hash:
            commit_tasks.append((repo_name, commit_hash))
    
    print(f"有效 commit 任务数: {len(commit_tasks)}")
    
    # 并行检查
    print(f"\n开始并行检查（{NUM_PROCESSES} 个进程）...")
    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(check_commit_directory, commit_tasks),
            total=len(commit_tasks),
            desc="检查 commit 目录"
        ))
    
    # 合并结果
    print("\n合并统计结果...")
    stats = merge_results(results)
    
    # 计算执行时间
    end_time = time.time()
    duration = end_time - start_time
    
    # 打印统计信息
    print("\n" + "="*80)
    print("检查完成")
    print("="*80)
    print(f"总 commit 数: {stats['total_commits']}")
    print(f"完整 (所有文件都存在且非空): {stats['complete']} ({stats['complete']/stats['total_commits']*100:.2f}%)")
    print(f"部分完整 (缺少部分文件): {stats['partial']} ({stats['partial']/stats['total_commits']*100:.2f}%)")
    print(f"不完整 (缺少大部分文件): {stats['incomplete']} ({stats['incomplete']/stats['total_commits']*100:.2f}%)")
    print(f"目标目录不存在: {stats['target_missing']} ({stats['target_missing']/stats['total_commits']*100:.2f}%)")
    print(f"源目录不存在: {stats['source_missing']} ({stats['source_missing']/stats['total_commits']*100:.2f}%)")
    print(f"\n文件数量统计:")
    print(f"  平均: {stats['file_count_stats']['avg']:.2f}")
    print(f"  最小: {stats['file_count_stats']['min']}")
    print(f"  最大: {stats['file_count_stats']['max']}")
    print(f"\n缺失文件统计:")
    for file_type, count in sorted(stats["missing_files_stats"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {file_type}: {count}")
    print(f"\n空文件统计:")
    for file_type, count in sorted(stats["empty_files_stats"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {file_type}: {count}")
    print(f"\n涉及代码库数: {len(stats['repo_stats'])}")
    print(f"执行时间: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    print("="*80)
    
    # 打印前10个代码库的统计
    if stats["repo_stats"]:
        print("\n前10个代码库统计:")
        sorted_repos = sorted(stats["repo_stats"].items(), 
                            key=lambda x: x[1]["total"], 
                            reverse=True)[:10]
        for repo_name, repo_stat in sorted_repos:
            complete_rate = repo_stat["complete"] / repo_stat["total"] * 100 if repo_stat["total"] > 0 else 0
            print(f"  {repo_name}: 总数={repo_stat['total']}, "
                  f"完整={repo_stat['complete']} ({complete_rate:.1f}%), "
                  f"部分={repo_stat['partial']}, "
                  f"不完整={repo_stat['incomplete']}, "
                  f"缺失={repo_stat['target_missing']}")
    
    # 打印问题摘要
    if stats["issues"]["target_missing"]:
        print(f"\n目标目录缺失的 commit 数量: {len(stats['issues']['target_missing'])}")
        print("示例（前5个）:")
        for issue in stats["issues"]["target_missing"][:5]:
            print(f"  {issue['repo']}/{issue['commit'][:10]}...")
    
    if stats["issues"]["missing_files"]:
        print(f"\n缺失文件的 commit 数量: {len(stats['issues']['missing_files'])}")
        print("示例（前5个）:")
        for issue in stats["issues"]["missing_files"][:5]:
            print(f"  {issue['repo']}/{issue['commit'][:10]}... 缺失: {issue['file_type']}")
    
    if stats["issues"]["empty_files"]:
        print(f"\n空文件的 commit 数量: {len(stats['issues']['empty_files'])}")
        print("示例（前5个）:")
        for issue in stats["issues"]["empty_files"][:5]:
            print(f"  {issue['repo']}/{issue['commit'][:10]}... 空文件: {issue['file_type']}")


if __name__ == "__main__":
    main()

