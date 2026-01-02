#!/usr/bin/env python3
"""
复制 all_one_func.json 中的 commits 对应的 semgrep 子文件夹
从 semopt_c_paper_backup/knowledge_base_all/ 中读取所有 commits，
检查是否存在 semgrep 子文件夹，如果存在则复制到 semopt_arch/knowledge_base/

目录结构：
- 原目录: knowledge_base_all/{repo_name}/modified_file/{commit_hash}/semgrep/
- 目标目录: knowledge_base/{repo_name}/modified_file/{commit_hash}/semgrep/
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
SOURCE_BASE = os.path.join(config.semopt_c_paper_backup_path, "knowledge_base_all")
TARGET_BASE = os.path.join(config.root_path, "knowledge_base")
ALL_ONE_FUNC_FILE = os.path.join(config.root_path, "all_one_func.json")

# 并行配置
NUM_PROCESSES = min(cpu_count(), 128)

# 是否跳过已存在的目录
SKIP_EXISTING = True


def check_and_copy_semgrep(args):
    """
    检查单个 commit 是否有 semgrep 文件夹，如果有则复制
    
    Args:
        args: (repo_name, commit_hash)
    
    Returns:
        dict: 处理结果统计
    """
    repo_name, commit_hash = args
    
    result = {
        "semgrep_found": 0,
        "semgrep_not_found": 0,
        "semgrep_copied": 0,
        "semgrep_skipped": 0,
        "files_copied": 0,
        "errors": []
    }
    
    try:
        # 构建源和目标路径
        source_semgrep_dir = os.path.join(
            SOURCE_BASE, repo_name, "modified_file", commit_hash, "semgrep"
        )
        target_semgrep_dir = os.path.join(
            TARGET_BASE, repo_name, "modified_file", commit_hash, "semgrep"
        )
        
        # 检查源 semgrep 文件夹是否存在
        if not os.path.exists(source_semgrep_dir):
            result["semgrep_not_found"] = 1
            return result
        
        # 检查是否是文件夹
        if not os.path.isdir(source_semgrep_dir):
            result["semgrep_not_found"] = 1
            return result
        
        result["semgrep_found"] = 1
        
        # 检查目标文件夹是否已存在
        if SKIP_EXISTING and os.path.exists(target_semgrep_dir):
            result["semgrep_skipped"] = 1
            # 统计已存在的文件数
            try:
                files = [f for f in os.listdir(target_semgrep_dir) 
                        if os.path.isfile(os.path.join(target_semgrep_dir, f))]
                result["files_copied"] = len(files)
            except Exception:
                pass
            return result
        
        # 复制 semgrep 文件夹
        try:
            # 确保目标父目录存在
            target_commit_dir = os.path.dirname(target_semgrep_dir)
            os.makedirs(target_commit_dir, exist_ok=True)
            
            # 复制整个文件夹
            shutil.copytree(source_semgrep_dir, target_semgrep_dir, dirs_exist_ok=False)
            
            result["semgrep_copied"] = 1
            
            # 统计复制的文件数
            files = [f for f in os.listdir(target_semgrep_dir) 
                    if os.path.isfile(os.path.join(target_semgrep_dir, f))]
            result["files_copied"] = len(files)
            
        except Exception as e:
            result["errors"].append({
                "repo": repo_name,
                "commit": commit_hash,
                "error": f"copy_failed: {str(e)}"
            })
    
    except Exception as e:
        result["errors"].append({
            "repo": repo_name,
            "commit": commit_hash,
            "error": f"unexpected_error: {str(e)}"
        })
    
    return result


def merge_results(results_list):
    """合并多个处理结果"""
    merged = {
        "semgrep_found": 0,
        "semgrep_not_found": 0,
        "semgrep_copied": 0,
        "semgrep_skipped": 0,
        "files_copied": 0,
        "errors": []
    }
    
    for result in results_list:
        merged["semgrep_found"] += result.get("semgrep_found", 0)
        merged["semgrep_not_found"] += result.get("semgrep_not_found", 0)
        merged["semgrep_copied"] += result.get("semgrep_copied", 0)
        merged["semgrep_skipped"] += result.get("semgrep_skipped", 0)
        merged["files_copied"] += result.get("files_copied", 0)
        merged["errors"].extend(result.get("errors", []))
    
    return merged


def load_commits(all_one_func_file):
    """
    从 all_one_func.json 中加载 commits
    
    Returns:
        list: [(repo_name, commit_hash), ...]
    """
    print("\n" + "="*80)
    print("加载 all_one_func.json")
    print("="*80)
    
    print(f"文件路径: {all_one_func_file}")
    
    try:
        with open(all_one_func_file, 'r', encoding='utf-8') as f:
            commits = json.load(f)
        
        print(f"✓ 成功加载 {len(commits)} 个 commits")
        
        # 提取 (repo_name, commit_hash) 对
        commit_pairs = []
        missing_info = 0
        
        for commit in commits:
            repo_name = commit.get("repository_name")
            commit_hash = commit.get("hash")
            
            if repo_name and commit_hash:
                commit_pairs.append((repo_name, commit_hash))
            else:
                missing_info += 1
        
        print(f"✓ 提取到 {len(commit_pairs)} 个有效的 (repo_name, commit_hash) 对")
        
        if missing_info > 0:
            print(f"⚠ {missing_info} 个 commits 缺少 repository_name 或 hash")
        
        # 去重
        unique_pairs = list(set(commit_pairs))
        duplicates = len(commit_pairs) - len(unique_pairs)
        
        if duplicates > 0:
            print(f"✓ 去重后剩余 {len(unique_pairs)} 个唯一的 commits（删除了 {duplicates} 个重复）")
        
        return unique_pairs
    
    except FileNotFoundError:
        print(f"✗ 错误：文件不存在 {all_one_func_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ 错误：JSON 解析失败 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 错误：{e}")
        sys.exit(1)


def process_all_commits(commit_pairs):
    """
    并行处理所有 commits
    """
    print("\n" + "="*80)
    print("检查并复制 semgrep 文件夹")
    print("="*80)
    
    print(f"总 commit 数: {len(commit_pairs)}")
    print(f"使用 {NUM_PROCESSES} 个并行进程")
    print(f"跳过已存在: {SKIP_EXISTING}")
    
    # 使用进程池并行处理
    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap_unordered(check_and_copy_semgrep, commit_pairs, chunksize=100),
            total=len(commit_pairs),
            desc="处理 commits"
        ))
    
    # 合并结果
    stats = merge_results(results)
    stats["total_commits"] = len(commit_pairs)
    
    return stats


def print_summary(stats, duration):
    """打印摘要信息"""
    print("\n" + "="*80)
    print("执行摘要")
    print("="*80)
    
    print(f"\n执行时间: {duration:.2f} 秒 (约 {duration/60:.2f} 分钟)")
    
    print("\n【处理结果】")
    print(f"  - 总 commit 数: {stats['total_commits']}")
    print(f"  - 找到 semgrep 文件夹: {stats['semgrep_found']}")
    print(f"  - 未找到 semgrep 文件夹: {stats['semgrep_not_found']}")
    print(f"  - 复制的 semgrep 文件夹: {stats['semgrep_copied']}")
    print(f"  - 跳过的 semgrep 文件夹: {stats['semgrep_skipped']}")
    print(f"  - 总文件数: {stats['files_copied']}")
    print(f"  - 错误数: {len(stats['errors'])}")
    
    # 计算百分比
    if stats['total_commits'] > 0:
        found_percentage = (stats['semgrep_found'] / stats['total_commits']) * 100
        print(f"\n【覆盖率】")
        print(f"  - semgrep 文件夹覆盖率: {found_percentage:.2f}%")
    
    # 显示错误详情（最多显示前10个）
    if stats['errors']:
        print(f"\n【错误详情】（显示前 10 个）")
        for i, error in enumerate(stats['errors'][:10], 1):
            print(f"  {i}. {error['repo']}/{error['commit']}: {error['error']}")
        
        if len(stats['errors']) > 10:
            print(f"  ... 还有 {len(stats['errors']) - 10} 个错误")
    
    print("\n" + "="*80)


def save_report(stats, output_file):
    """保存详细报告到 JSON 文件"""
    report = {
        "summary": {
            "total_commits": stats["total_commits"],
            "semgrep_found": stats["semgrep_found"],
            "semgrep_not_found": stats["semgrep_not_found"],
            "semgrep_copied": stats["semgrep_copied"],
            "semgrep_skipped": stats["semgrep_skipped"],
            "total_files_copied": stats["files_copied"],
            "error_count": len(stats["errors"])
        },
        "errors": stats["errors"]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 详细报告已保存到: {output_file}")


def main():
    """主函数"""
    start_time = time.time()
    
    print("="*80)
    print("复制 all_one_func.json 中 commits 的 semgrep 文件夹")
    print("="*80)
    print(f"源基础目录: {SOURCE_BASE}")
    print(f"目标基础目录: {TARGET_BASE}")
    print(f"all_one_func.json: {ALL_ONE_FUNC_FILE}")
    print(f"CPU 核心数: {cpu_count()}")
    
    # 检查源目录是否存在
    if not os.path.exists(SOURCE_BASE):
        print(f"\n✗ 错误：源目录不存在 {SOURCE_BASE}")
        sys.exit(1)
    
    # 检查 all_one_func.json 是否存在
    if not os.path.exists(ALL_ONE_FUNC_FILE):
        print(f"\n✗ 错误：all_one_func.json 不存在 {ALL_ONE_FUNC_FILE}")
        sys.exit(1)
    
    # 创建目标目录
    os.makedirs(TARGET_BASE, exist_ok=True)
    
    # 加载 commits
    commit_pairs = load_commits(ALL_ONE_FUNC_FILE)
    
    # 处理所有 commits
    stats = process_all_commits(commit_pairs)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 打印摘要
    print_summary(stats, duration)
    
    print("\n✓ 所有操作完成！")


if __name__ == "__main__":
    main()

