#!/usr/bin/env python3
"""
复制知识库文件脚本
从 semopt_c_paper_backup/knowledge_base_all/ 复制文件到 semopt_arch/knowledge_base/

包括两部分：
1. JSON 文件：为 repo_list_30342.json 中的所有代码库复制对应的 JSON 文件
2. Commit 详细信息：为 all_is_opt_final.json 中的所有 commit 复制对应的详细文件
"""

import json
import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

# 添加 python/ 目录到 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 路径配置（从 config 中获取）
SOURCE_BASE = os.path.join(config.semopt_c_paper_backup_path, "knowledge_base_all")
TARGET_BASE = os.path.join(config.root_path, "knowledge_base")
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
COMMIT_LIST_FILE = os.path.join(config.semopt_c_paper_backup_path, "all_is_opt_final.json")

# 需要复制的 JSON 文件列表（严格按照技术方案，不包含可选/旧版文件）
JSON_FILES_TO_COPY = [
    "all_commit.json",
    "one_file.json",
    "c_language.json",
    "is_opt_keyword.json",
    "has_file.json",
    "has_file_deduplicate.json",
    "diff.json",
    "one_func.json",
    "line_block.json",
    "func_name.json",
    "func_name_result.json",
    "is_opt_llm.json",
    "is_opt_final.json",
    "summary.json",
    "summary_filter.json",
]

# 需要复制的 commit 文件类型
COMMIT_FILE_PATTERNS = {
    "diff.txt": "diff.txt",
    "before": "before.*",
    "after": "after.*",
    "before_func": "before_func.*",
    "after_func": "after_func.*",
}

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
        print(f"Error reading directory {commit_dir}: {e}")
        return found_files
    
    # 查找 diff.txt
    if "diff.txt" in all_files:
        found_files["diff.txt"] = "diff.txt"
    
    # 查找 before.* (排除 before_func.*)
    for f in all_files:
        if f.startswith("before.") and not should_exclude_file(f):
            found_files["before"] = f
            break
    
    # 查找 after.* (排除 after_func.*)
    for f in all_files:
        if f.startswith("after.") and not should_exclude_file(f):
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


def copy_json_files_for_repo(repo_name):
    """
    为单个代码库复制 JSON 文件
    返回该代码库的统计信息
    """
    result = {
        "success": 0,
        "source_not_exist": 0,
        "no_json_files": 0,
        "files_copied": 0,
        "errors": [],
    }
    
    source_repo_dir = os.path.join(SOURCE_BASE, repo_name)
    target_repo_dir = os.path.join(TARGET_BASE, repo_name)
    
    # 检查源目录是否存在
    if not os.path.exists(source_repo_dir):
        result["source_not_exist"] = 1
        return result
    
    # 创建目标目录
    os.makedirs(target_repo_dir, exist_ok=True)
    
    # 复制 JSON 文件
    files_copied_for_repo = 0
    for json_file in JSON_FILES_TO_COPY:
        source_file = os.path.join(source_repo_dir, json_file)
        target_file = os.path.join(target_repo_dir, json_file)
        
        if os.path.exists(source_file):
            try:
                shutil.copy2(source_file, target_file)
                files_copied_for_repo += 1
                result["files_copied"] += 1
            except Exception as e:
                result["errors"].append({
                    "repo": repo_name,
                    "file": json_file,
                    "error": str(e)
                })
    
    if files_copied_for_repo > 0:
        result["success"] = 1
    else:
        result["no_json_files"] = 1
    
    return result


def merge_stats(stats_list):
    """合并多个统计结果"""
    merged = {
        "success": 0,
        "source_not_exist": 0,
        "no_json_files": 0,
        "files_copied": 0,
        "errors": [],
    }
    
    for stats in stats_list:
        merged["success"] += stats.get("success", 0)
        merged["source_not_exist"] += stats.get("source_not_exist", 0)
        merged["no_json_files"] += stats.get("no_json_files", 0)
        merged["files_copied"] += stats.get("files_copied", 0)
        merged["errors"].extend(stats.get("errors", []))
    
    return merged


def copy_json_files(repo_list, num_processes=None, quiet=False):
    """
    阶段一：复制 JSON 文件（并行）
    """
    if not quiet:
        print("\n" + "="*80)
        print("阶段一：复制 JSON 文件")
        print("="*80)
    
    if num_processes is None:
        num_processes = min(cpu_count(), 128)  # 最多使用128个进程
    
    if not quiet:
        print(f"使用 {num_processes} 个并行进程")
    
    # 提取代码库名称列表
    repo_names = [repo_info["name"] for repo_info in repo_list]
    
    # 使用进程池并行处理
    with Pool(processes=num_processes) as pool:
        if quiet:
            results = list(pool.imap(copy_json_files_for_repo, repo_names))
        else:
            results = list(tqdm(
                pool.imap(copy_json_files_for_repo, repo_names),
                total=len(repo_names),
                desc="复制 JSON 文件"
            ))
    
    # 合并统计结果
    stats = merge_stats(results)
    stats["total_repos"] = len(repo_list)
    
    return stats


def copy_commit_files_for_repo(args):
    """
    为单个代码库的所有 commits 复制文件
    返回该代码库的统计信息
    """
    repo_name, commit_hashes = args
    
    result = {
        "success_commits": 0,
        "source_not_exist": 0,
        "partial_success": 0,
        "files_copied": 0,
        "missing_files": defaultdict(int),
        "errors": [],
    }
    
    for commit_hash in commit_hashes:
        source_commit_dir = os.path.join(SOURCE_BASE, repo_name, "modified_file", commit_hash)
        target_commit_dir = os.path.join(TARGET_BASE, repo_name, "modified_file", commit_hash)
        
        # 检查源目录是否存在
        if not os.path.exists(source_commit_dir):
            result["source_not_exist"] += 1
            continue
        
        # 创建目标目录
        os.makedirs(target_commit_dir, exist_ok=True)
        
        # 查找需要的文件
        found_files = find_commit_files(source_commit_dir)
        
        # 复制文件
        files_copied_for_commit = 0
        expected_files = ["diff.txt", "before", "after", "before_func", "after_func"]
        
        for file_type in expected_files:
            if file_type in found_files:
                filename = found_files[file_type]
                source_file = os.path.join(source_commit_dir, filename)
                target_file = os.path.join(target_commit_dir, filename)
                
                try:
                    # 只复制文件，不复制目录
                    if os.path.isfile(source_file):
                        shutil.copy2(source_file, target_file)
                        files_copied_for_commit += 1
                        result["files_copied"] += 1
                except Exception as e:
                    result["errors"].append({
                        "repo": repo_name,
                        "commit": commit_hash,
                        "file": filename,
                        "error": str(e)
                    })
            else:
                result["missing_files"][file_type] += 1
        
        # 统计成功情况
        if files_copied_for_commit == 5:
            result["success_commits"] += 1
        elif files_copied_for_commit > 0:
            result["partial_success"] += 1
    
    return result


def merge_commit_stats(stats_list):
    """合并多个 commit 统计结果"""
    merged = {
        "success_commits": 0,
        "source_not_exist": 0,
        "partial_success": 0,
        "files_copied": 0,
        "missing_files": defaultdict(int),
        "errors": [],
    }
    
    for stats in stats_list:
        merged["success_commits"] += stats.get("success_commits", 0)
        merged["source_not_exist"] += stats.get("source_not_exist", 0)
        merged["partial_success"] += stats.get("partial_success", 0)
        merged["files_copied"] += stats.get("files_copied", 0)
        
        # 合并 missing_files
        for file_type, count in stats.get("missing_files", {}).items():
            merged["missing_files"][file_type] += count
        
        merged["errors"].extend(stats.get("errors", []))
    
    return merged


def copy_commit_files(commit_list, num_processes=None, quiet=False):
    """
    阶段二：复制 Commit 详细信息文件（并行）
    """
    if not quiet:
        print("\n" + "="*80)
        print("阶段二：复制 Commit 详细信息文件")
        print("="*80)
    
    # 按代码库分组 commits
    commits_by_repo = defaultdict(list)
    for commit in commit_list:
        repo_name = commit["repository_name"]
        commit_hash = commit["hash"]
        commits_by_repo[repo_name].append(commit_hash)
    
    if not quiet:
        print(f"总计 {len(commit_list)} 个 commits，分布在 {len(commits_by_repo)} 个代码库中")
    
    if num_processes is None:
        num_processes = min(cpu_count(), 128)  # 最多使用128个进程
    
    if not quiet:
        print(f"使用 {num_processes} 个并行进程")
    
    # 准备参数列表
    repo_args = list(commits_by_repo.items())
    
    # 使用进程池并行处理
    with Pool(processes=num_processes) as pool:
        if quiet:
            results = list(pool.imap(copy_commit_files_for_repo, repo_args))
        else:
            results = list(tqdm(
                pool.imap(copy_commit_files_for_repo, repo_args),
                total=len(repo_args),
                desc="处理代码库 commits"
            ))
    
    # 合并统计结果
    stats = merge_commit_stats(results)
    stats["total_commits"] = len(commit_list)
    stats["total_repos"] = len(commits_by_repo)
    
    return stats


def generate_report(json_stats, commit_stats, start_time, end_time):
    """生成详细报告"""
    report = {
        "execution_time": {
            "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            "duration_seconds": round(end_time - start_time, 2),
        },
        "json_files_copy": {
            "total_repos": json_stats["total_repos"],
            "success_repos": json_stats["success"],
            "source_not_exist": json_stats["source_not_exist"],
            "no_json_files": json_stats["no_json_files"],
            "total_files_copied": json_stats["files_copied"],
            "errors_count": len(json_stats["errors"]),
            "errors": json_stats["errors"][:10] if json_stats["errors"] else [],  # 只保存前10个错误
        },
        "commit_files_copy": {
            "total_commits": commit_stats["total_commits"],
            "total_repos": commit_stats["total_repos"],
            "success_commits": commit_stats["success_commits"],
            "partial_success_commits": commit_stats["partial_success"],
            "source_not_exist": commit_stats["source_not_exist"],
            "total_files_copied": commit_stats["files_copied"],
            "missing_files_by_type": dict(commit_stats["missing_files"]),
            "errors_count": len(commit_stats["errors"]),
            "errors": commit_stats["errors"][:10] if commit_stats["errors"] else [],  # 只保存前10个错误
        },
    }
    
    return report


def print_summary(report):
    """打印摘要信息"""
    print("\n" + "="*80)
    print("执行摘要")
    print("="*80)
    
    print(f"\n执行时间: {report['execution_time']['duration_seconds']} 秒")
    
    print("\n【JSON 文件复制】")
    json_copy = report["json_files_copy"]
    print(f"  - 总代码库数: {json_copy['total_repos']}")
    print(f"  - 成功复制: {json_copy['success_repos']}")
    print(f"  - 源不存在: {json_copy['source_not_exist']}")
    print(f"  - 无JSON文件: {json_copy['no_json_files']}")
    print(f"  - 总文件数: {json_copy['total_files_copied']}")
    print(f"  - 错误数: {json_copy['errors_count']}")
    
    print("\n【Commit 文件复制】")
    commit_copy = report["commit_files_copy"]
    print(f"  - 总 Commit 数: {commit_copy['total_commits']}")
    print(f"  - 涉及代码库: {commit_copy['total_repos']}")
    print(f"  - 完整复制: {commit_copy['success_commits']}")
    print(f"  - 部分复制: {commit_copy['partial_success_commits']}")
    print(f"  - 源不存在: {commit_copy['source_not_exist']}")
    print(f"  - 总文件数: {commit_copy['total_files_copied']}")
    print(f"  - 错误数: {commit_copy['errors_count']}")
    
    print("\n【文件缺失统计】")
    for file_type, count in commit_copy["missing_files_by_type"].items():
        print(f"  - {file_type}: {count} 个 commit")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='复制知识库文件脚本')
    parser.add_argument('--quiet', '-q', action='store_true', 
                        help='静默模式，只输出最少的信息')
    args = parser.parse_args()
    
    start_time = time.time()
    
    if not args.quiet:
        print("="*80)
        print("知识库文件复制脚本")
        print("="*80)
        print(f"源目录: {SOURCE_BASE}")
        print(f"目标目录: {TARGET_BASE}")
        print(f"代码库列表: {REPO_LIST_FILE}")
        print(f"Commit 列表: {COMMIT_LIST_FILE}")
        print(f"CPU 核心数: {cpu_count()}")
    
    # 读取代码库列表
    if not args.quiet:
        print("\n正在读取代码库列表...")
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repo_list = json.load(f)
    if not args.quiet:
        print(f"✓ 读取到 {len(repo_list)} 个代码库")
    
    # 读取 commit 列表
    if not args.quiet:
        print("\n正在读取 commit 列表...")
    with open(COMMIT_LIST_FILE, 'r', encoding='utf-8') as f:
        commit_list = json.load(f)
    if not args.quiet:
        print(f"✓ 读取到 {len(commit_list)} 个 commits")
    
    # 创建目标基础目录
    os.makedirs(TARGET_BASE, exist_ok=True)
    
    # 阶段一：复制 JSON 文件（并行）
    json_stats = copy_json_files(repo_list, num_processes=128, quiet=args.quiet)
    
    # 阶段二：复制 Commit 文件（并行）
    commit_stats = copy_commit_files(commit_list, num_processes=128, quiet=args.quiet)
    
    end_time = time.time()
    
    # 生成报告（仅用于打印摘要，不保存文件）
    report = generate_report(json_stats, commit_stats, start_time, end_time)
    
    # 打印摘要
    if not args.quiet:
        print_summary(report)
        print("\n✓ 所有操作完成！")
    else:
        # 静默模式下只输出一行结果
        print(f"完成: JSON文件 {json_stats['files_copied']}, Commit文件 {commit_stats['files_copied']}, 耗时 {report['execution_time']['duration_seconds']}秒")


if __name__ == "__main__":
    main()

