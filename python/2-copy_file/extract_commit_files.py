#!/usr/bin/env python3
"""
提取 commit 详细信息文件夹脚本
从 knowledge_base/ 中提取 all_is_opt_final.json 中所有 commit 的详细信息文件夹
复制到 semopt_commit_list/knowledge_base/ 目录下，保持原有目录结构
"""

import json
import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# 添加 python/ 目录到 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
SOURCE_KNOWLEDGE_BASE = os.path.join(config.root_path, "knowledge_base")
TARGET_KNOWLEDGE_BASE = os.path.join(config.root_path, "semopt_commit_list", "knowledge_base")
COMMIT_LIST_FILE = os.path.join(config.root_path, "all_is_opt_final.json")

# 并行处理配置
NUM_PROCESSES = min(cpu_count(), 128)  # 最多使用128个进程

# 是否跳过已存在的目录
SKIP_EXISTING = True


def copy_commit_directory(args):
    """
    复制单个 commit 的详细信息目录
    
    Args:
        args: (repo_name, commit_hash) 元组
    
    Returns:
        dict: 处理结果
    """
    repo_name, commit_hash = args
    
    result = {
        "repo_name": repo_name,
        "commit_hash": commit_hash,
        "status": "unknown",
        "error": None,
        "files_copied": 0,
    }
    
    try:
        # 构建源路径和目标路径
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
        
        # 检查源目录是否存在
        if not os.path.exists(source_commit_dir):
            result["status"] = "source_not_exist"
            return result
        
        # 检查目标目录是否已存在
        if SKIP_EXISTING and os.path.exists(target_commit_dir):
            result["status"] = "skipped"
            # 统计已存在目录中的文件数
            if os.path.isdir(target_commit_dir):
                result["files_copied"] = len([f for f in os.listdir(target_commit_dir) 
                                            if os.path.isfile(os.path.join(target_commit_dir, f))])
            return result
        
        # 创建目标目录的父目录
        os.makedirs(os.path.dirname(target_commit_dir), exist_ok=True)
        
        # 复制整个目录
        if os.path.isdir(source_commit_dir):
            # 使用 copytree 复制整个目录
            shutil.copytree(source_commit_dir, target_commit_dir, dirs_exist_ok=True)
            
            # 统计复制的文件数
            result["files_copied"] = len([f for f in os.listdir(target_commit_dir) 
                                         if os.path.isfile(os.path.join(target_commit_dir, f))])
            result["status"] = "success"
        else:
            result["status"] = "source_not_dir"
            result["error"] = f"Source path is not a directory: {source_commit_dir}"
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def merge_results(results):
    """合并处理结果"""
    merged = {
        "total_commits": len(results),
        "success": 0,
        "skipped": 0,
        "source_not_exist": 0,
        "source_not_dir": 0,
        "errors": 0,
        "total_files_copied": 0,
        "error_list": [],
        "repo_stats": defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "skipped": 0,
            "source_not_exist": 0,
            "errors": 0,
        }),
    }
    
    for result in results:
        status = result["status"]
        repo_name = result["repo_name"]
        
        merged["repo_stats"][repo_name]["total"] += 1
        
        if status == "success":
            merged["success"] += 1
            merged["repo_stats"][repo_name]["success"] += 1
            merged["total_files_copied"] += result["files_copied"]
        elif status == "skipped":
            merged["skipped"] += 1
            merged["repo_stats"][repo_name]["skipped"] += 1
            merged["total_files_copied"] += result["files_copied"]
        elif status == "source_not_exist":
            merged["source_not_exist"] += 1
            merged["repo_stats"][repo_name]["source_not_exist"] += 1
        elif status == "source_not_dir":
            merged["source_not_dir"] += 1
            merged["repo_stats"][repo_name]["errors"] += 1
        elif status == "error":
            merged["errors"] += 1
            merged["repo_stats"][repo_name]["errors"] += 1
            merged["error_list"].append(result)
    
    return merged


def main():
    """主函数"""
    print("="*80)
    print("提取 Commit 详细信息文件夹")
    print("="*80)
    print(f"源目录: {SOURCE_KNOWLEDGE_BASE}")
    print(f"目标目录: {TARGET_KNOWLEDGE_BASE}")
    print(f"Commit 列表文件: {COMMIT_LIST_FILE}")
    print(f"并行进程数: {NUM_PROCESSES}")
    print(f"跳过已存在: {SKIP_EXISTING}")
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
    
    # 确保目标目录存在
    os.makedirs(TARGET_KNOWLEDGE_BASE, exist_ok=True)
    
    # 并行处理
    print(f"\n开始并行处理（{NUM_PROCESSES} 个进程）...")
    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(copy_commit_directory, commit_tasks),
            total=len(commit_tasks),
            desc="复制 commit 目录"
        ))
    
    # 合并结果
    print("\n合并统计结果...")
    stats = merge_results(results)
    
    # 计算执行时间
    end_time = time.time()
    duration = end_time - start_time
    
    # 打印统计信息
    print("\n" + "="*80)
    print("执行完成")
    print("="*80)
    print(f"总 commit 数: {stats['total_commits']}")
    print(f"成功复制: {stats['success']}")
    print(f"已跳过: {stats['skipped']}")
    print(f"源目录不存在: {stats['source_not_exist']}")
    print(f"源路径不是目录: {stats['source_not_dir']}")
    print(f"错误数: {stats['errors']}")
    print(f"总文件数: {stats['total_files_copied']}")
    print(f"涉及代码库数: {len(stats['repo_stats'])}")
    print(f"执行时间: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    print("="*80)
    
    # 打印前10个代码库的统计
    if stats["repo_stats"]:
        print("\n前10个代码库统计:")
        sorted_repos = sorted(stats["repo_stats"].items(), 
                            key=lambda x: x[1]["total"], 
                            reverse=True)[:10]
        for repo_name, repo_stat in sorted_repos:
            print(f"  {repo_name}: 总数={repo_stat['total']}, "
                  f"成功={repo_stat['success']}, "
                  f"跳过={repo_stat['skipped']}, "
                  f"不存在={repo_stat['source_not_exist']}, "
                  f"错误={repo_stat['errors']}")


if __name__ == "__main__":
    main()

