#!/usr/bin/env python3
"""
复制 one_func.json 中的 commits 详细文件
从 semopt_c_paper_backup/knowledge_base_all/ 中读取所有代码库的 one_func.json，
提取其中的 commit 信息，并将这些 commit 对应的详细文件复制到 semopt_arch/knowledge_base/

包括：
1. 读取所有代码库的 one_func.json
2. 汇总并去重 commits
3. 复制 commit 详细文件（diff.txt, before.*, after.*, before_func.*, after_func.*）
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# 添加 python/ 目录到 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
SOURCE_BASE = os.path.join(config.semopt_c_paper_backup_path, "knowledge_base_all")
TARGET_BASE = os.path.join(config.root_path, "knowledge_base")
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
OUTPUT_JSON = os.path.join(config.root_path, "all_one_func.json")

# 可选：与已有数据对比
EXISTING_COMMITS_FILE = os.path.join(config.root_path, "all_is_opt_final.json")

# 并行配置
NUM_PROCESSES_READ = min(cpu_count(), 128)  # 读取 JSON 阶段
NUM_PROCESSES_COPY = min(cpu_count(), 128)  # 复制文件阶段

# 是否跳过已存在的目录
SKIP_EXISTING = True

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


def read_one_func_from_repo(repo_name):
    """
    读取单个代码库的 one_func.json
    返回: {"success": True/False, "commits": [...], "error": None}
    """
    result = {
        "success": False,
        "repo_name": repo_name,
        "commits": [],
        "error": None,
        "commit_count": 0,
    }
    
    source_repo_dir = os.path.join(SOURCE_BASE, repo_name)
    one_func_file = os.path.join(source_repo_dir, "one_func.json")
    
    # 检查文件是否存在
    if not os.path.exists(one_func_file):
        result["error"] = "file_not_exist"
        return result
    
    try:
        # 读取 JSON 文件
        with open(one_func_file, 'r', encoding='utf-8') as f:
            commits = json.load(f)
        
        # 为每个 commit 添加 repository_name 字段
        for commit in commits:
            commit_with_repo = {"repository_name": repo_name}
            commit_with_repo.update(commit)
            result["commits"].append(commit_with_repo)
        
        result["success"] = True
        result["commit_count"] = len(commits)
    
    except json.JSONDecodeError as e:
        result["error"] = f"json_decode_error: {str(e)}"
    except Exception as e:
        result["error"] = f"unexpected_error: {str(e)}"
    
    return result


def collect_all_one_func_commits(repo_list):
    """
    并行读取所有代码库的 one_func.json
    """
    print("\n" + "="*80)
    print("阶段一：收集所有代码库的 one_func.json")
    print("="*80)
    
    repo_names = [repo_info["name"] for repo_info in repo_list]
    print(f"总代码库数: {len(repo_names)}")
    print(f"使用 {NUM_PROCESSES_READ} 个线程并行读取")
    
    all_commits = []
    stats = {
        "total_repos": len(repo_names),
        "success_repos": 0,
        "missing_json_files": 0,
        "json_decode_errors": 0,
        "unexpected_errors": 0,
        "total_commits": 0,
        "errors": [],
    }
    
    # 使用线程池并行读取（I/O密集）
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES_READ) as executor:
        futures = {executor.submit(read_one_func_from_repo, name): name for name in repo_names}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="读取 one_func.json"):
            result = future.result()
            
            if result["success"]:
                stats["success_repos"] += 1
                stats["total_commits"] += result["commit_count"]
                all_commits.extend(result["commits"])
            else:
                if result["error"] == "file_not_exist":
                    stats["missing_json_files"] += 1
                elif "json_decode_error" in result["error"]:
                    stats["json_decode_errors"] += 1
                    stats["errors"].append({
                        "repo": result["repo_name"],
                        "error": result["error"]
                    })
                else:
                    stats["unexpected_errors"] += 1
                    stats["errors"].append({
                        "repo": result["repo_name"],
                        "error": result["error"]
                    })
    
    print(f"\n收集完成:")
    print(f"  - 成功读取: {stats['success_repos']}")
    print(f"  - 文件不存在: {stats['missing_json_files']}")
    print(f"  - JSON 解析错误: {stats['json_decode_errors']}")
    print(f"  - 其他错误: {stats['unexpected_errors']}")
    print(f"  - 总 commit 数: {stats['total_commits']}")
    
    return all_commits, stats


def deduplicate_commits(commits, existing_commits=None):
    """
    去重 commits
    """
    print("\n" + "="*80)
    print("阶段二：去重处理")
    print("="*80)
    
    stats = {
        "commits_before_dedup": len(commits),
        "commits_after_dedup": 0,
        "duplicates_removed": 0,
        "already_exists_in_target": 0,
        "commits_to_copy": 0,
    }
    
    # 按 (repository_name, hash) 去重
    print("按 (repository_name, hash) 去重...")
    unique_commits_dict = {}
    
    for commit in tqdm(commits, desc="去重处理"):
        dedup_key = (commit.get("repository_name"), commit.get("hash"))
        if dedup_key not in unique_commits_dict:
            unique_commits_dict[dedup_key] = commit
    
    unique_commits = list(unique_commits_dict.values())
    stats["commits_after_dedup"] = len(unique_commits)
    stats["duplicates_removed"] = stats["commits_before_dedup"] - stats["commits_after_dedup"]
    
    print(f"  - 去重前: {stats['commits_before_dedup']}")
    print(f"  - 去重后: {stats['commits_after_dedup']}")
    print(f"  - 重复删除: {stats['duplicates_removed']}")
    
    # 与已有数据对比（可选）
    commits_to_copy = unique_commits
    if existing_commits and os.path.exists(EXISTING_COMMITS_FILE):
        print(f"\n与已有数据对比 ({EXISTING_COMMITS_FILE})...")
        try:
            with open(EXISTING_COMMITS_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            
            existing_set = set()
            for ec in existing:
                key = (ec.get("repository_name"), ec.get("hash"))
                existing_set.add(key)
            
            # 过滤掉已存在的
            commits_to_copy = []
            for commit in unique_commits:
                key = (commit.get("repository_name"), commit.get("hash"))
                if key not in existing_set:
                    commits_to_copy.append(commit)
                else:
                    stats["already_exists_in_target"] += 1
            
            stats["commits_to_copy"] = len(commits_to_copy)
            print(f"  - 已存在于目标: {stats['already_exists_in_target']}")
            print(f"  - 需要新复制: {stats['commits_to_copy']}")
        
        except Exception as e:
            print(f"  - 读取已有数据失败: {e}")
            print(f"  - 将复制所有 commits")
            commits_to_copy = unique_commits
            stats["commits_to_copy"] = len(commits_to_copy)
    else:
        stats["commits_to_copy"] = len(commits_to_copy)
    
    return unique_commits, commits_to_copy, stats


def group_commits_by_repo(commits):
    """
    按代码库分组 commits
    """
    print("\n" + "="*80)
    print("阶段三：按代码库分组")
    print("="*80)
    
    commits_by_repo = defaultdict(list)
    for commit in commits:
        repo_name = commit.get("repository_name")
        commit_hash = commit.get("hash")
        if repo_name and commit_hash:
            commits_by_repo[repo_name].append(commit_hash)
    
    print(f"涉及代码库数: {len(commits_by_repo)}")
    print(f"总 commit 数: {sum(len(v) for v in commits_by_repo.values())}")
    
    # 显示前 10 个代码库的统计
    sorted_repos = sorted(commits_by_repo.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print("\n前 10 个代码库的 commit 数量:")
    for repo_name, commit_hashes in sorted_repos:
        print(f"  - {repo_name}: {len(commit_hashes)} commits")
    
    return commits_by_repo


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
        "skipped": 0,
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
        
        # 检查目标目录是否已存在
        if SKIP_EXISTING and os.path.exists(target_commit_dir):
            result["skipped"] += 1
            # 统计已存在目录中的文件数
            try:
                files = [f for f in os.listdir(target_commit_dir) 
                        if os.path.isfile(os.path.join(target_commit_dir, f))]
                result["files_copied"] += len(files)
            except Exception:
                pass
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
        "skipped": 0,
        "files_copied": 0,
        "missing_files": defaultdict(int),
        "errors": [],
    }
    
    for stats in stats_list:
        merged["success_commits"] += stats.get("success_commits", 0)
        merged["source_not_exist"] += stats.get("source_not_exist", 0)
        merged["partial_success"] += stats.get("partial_success", 0)
        merged["skipped"] += stats.get("skipped", 0)
        merged["files_copied"] += stats.get("files_copied", 0)
        
        # 合并 missing_files
        for file_type, count in stats.get("missing_files", {}).items():
            merged["missing_files"][file_type] += count
        
        merged["errors"].extend(stats.get("errors", []))
    
    return merged


def copy_all_commits(commits_by_repo):
    """
    并行复制所有代码库的 commits
    """
    print("\n" + "="*80)
    print("阶段四：复制 Commit 详细文件")
    print("="*80)
    
    print(f"使用 {NUM_PROCESSES_COPY} 个并行进程")
    
    # 准备参数列表
    repo_args = list(commits_by_repo.items())
    
    # 使用进程池并行处理
    with Pool(processes=NUM_PROCESSES_COPY) as pool:
        results = list(tqdm(
            pool.imap(copy_commit_files_for_repo, repo_args),
            total=len(repo_args),
            desc="复制 commit 文件"
        ))
    
    # 合并统计结果
    stats = merge_commit_stats(results)
    stats["total_commits"] = sum(len(hashes) for hashes in commits_by_repo.values())
    stats["total_repos"] = len(commits_by_repo)
    
    return stats


def print_summary(collection_stats, dedup_stats, copy_stats, duration):
    """打印摘要信息"""
    print("\n" + "="*80)
    print("执行摘要")
    print("="*80)
    
    print(f"\n执行时间: {duration:.2f} 秒")
    
    print("\n【阶段一：收集 one_func.json】")
    print(f"  - 总代码库数: {collection_stats['total_repos']}")
    print(f"  - 成功读取: {collection_stats['success_repos']}")
    print(f"  - 文件不存在: {collection_stats['missing_json_files']}")
    print(f"  - JSON解析错误: {collection_stats['json_decode_errors']}")
    print(f"  - 其他错误: {collection_stats['unexpected_errors']}")
    print(f"  - 总 commit 数: {collection_stats['total_commits']}")
    print(f"  - 错误数: {len(collection_stats['errors'])}")
    
    print("\n【阶段二：去重处理】")
    print(f"  - 去重前: {dedup_stats['commits_before_dedup']}")
    print(f"  - 去重后: {dedup_stats['commits_after_dedup']}")
    print(f"  - 重复删除: {dedup_stats['duplicates_removed']}")
    print(f"  - 已存在于目标: {dedup_stats['already_exists_in_target']}")
    print(f"  - 需要新复制: {dedup_stats['commits_to_copy']}")
    
    print("\n【阶段三：复制文件】")
    print(f"  - 总 Commit 数: {copy_stats['total_commits']}")
    print(f"  - 涉及代码库: {copy_stats['total_repos']}")
    print(f"  - 完整复制: {copy_stats['success_commits']}")
    print(f"  - 部分复制: {copy_stats['partial_success']}")
    print(f"  - 源不存在: {copy_stats['source_not_exist']}")
    print(f"  - 已跳过: {copy_stats['skipped']}")
    print(f"  - 总文件数: {copy_stats['files_copied']}")
    print(f"  - 错误数: {len(copy_stats['errors'])}")
    
    if copy_stats["missing_files"]:
        print("\n【文件缺失统计】")
        for file_type, count in sorted(copy_stats["missing_files"].items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"  - {file_type}: {count} 个 commit")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    start_time = time.time()
    
    print("="*80)
    print("复制 one_func.json 中的 Commits 详细文件")
    print("="*80)
    print(f"源目录: {SOURCE_BASE}")
    print(f"目标目录: {TARGET_BASE}")
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"输出 JSON: {OUTPUT_JSON}")
    print(f"跳过已存在: {SKIP_EXISTING}")
    print(f"CPU 核心数: {cpu_count()}")
    
    # 读取代码库列表
    print("\n正在读取代码库列表...")
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repo_list = json.load(f)
    print(f"✓ 读取到 {len(repo_list)} 个代码库")
    
    # 创建目标基础目录
    os.makedirs(TARGET_BASE, exist_ok=True)
    
    # 阶段一：收集所有 one_func.json
    all_commits, collection_stats = collect_all_one_func_commits(repo_list)
    
    # 保存汇总的 commits
    print(f"\n保存汇总结果到 {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_commits, f, indent=2, ensure_ascii=False)
    print("✓ 汇总结果已保存")
    
    # 阶段二：去重处理
    unique_commits, commits_to_copy, dedup_stats = deduplicate_commits(
        all_commits, 
        existing_commits=True
    )
    
    # 阶段三：按代码库分组
    commits_by_repo = group_commits_by_repo(commits_to_copy)
    
    # 阶段四：复制 commit 文件
    copy_stats = copy_all_commits(commits_by_repo)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 打印摘要
    print_summary(collection_stats, dedup_stats, copy_stats, duration)
    
    print("\n✓ 所有操作完成！")


if __name__ == "__main__":
    main()

