#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 knowledge_base 中提取 all_is_opt_final.json 涉及的代码库数据
并复制到 huawei_stage2/knowledge_base 中
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def load_all_commits(json_path: str) -> List[Dict]:
    """加载 all_is_opt_final.json 文件"""
    print(f"正在读取 {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        commits = json.load(f)
    print(f"成功读取 {len(commits)} 个 commit")
    return commits


def group_commits_by_repository(commits: List[Dict]) -> Dict[str, List[Dict]]:
    """按代码库名称对 commit 进行分组"""
    print("\n正在按代码库分组...")
    grouped = defaultdict(list)
    
    for commit in commits:
        repo_name = commit.get('repository_name', 'unknown')
        grouped[repo_name].append(commit)
    
    print(f"共发现 {len(grouped)} 个不同的代码库:")
    for repo_name, repo_commits in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  - {repo_name}: {len(repo_commits)} 个 commit")
    if len(grouped) > 10:
        print(f"  ... 以及其他 {len(grouped) - 10} 个代码库")
    
    return dict(grouped)


def create_repository_directories(target_base: Path, repo_names: List[str]) -> None:
    """创建目标代码库目录"""
    print("\n正在创建目标目录结构...")
    for repo_name in repo_names:
        repo_dir = target_base / repo_name
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建 modified_file 子目录
        modified_file_dir = repo_dir / "modified_file"
        modified_file_dir.mkdir(exist_ok=True)
    
    print(f"成功创建 {len(repo_names)} 个代码库目录")


def save_repository_json(target_base: Path, repo_name: str, commits: List[Dict]) -> None:
    """保存代码库的 is_opt_final.json 文件"""
    json_path = target_base / repo_name / "is_opt_final.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(commits, f, ensure_ascii=False, indent=2)


def copy_commit_files(source_base: Path, target_base: Path, repo_name: str, 
                     commit_hashes: Set[str]) -> tuple:
    """复制 commit 的 modified_file 目录"""
    success_count = 0
    missing_count = 0
    error_count = 0
    
    source_repo = source_base / repo_name
    target_repo = target_base / repo_name
    
    # 检查源代码库目录是否存在
    if not source_repo.exists():
        print(f"  [警告] 源代码库目录不存在: {source_repo}")
        return success_count, len(commit_hashes), error_count
    
    source_modified_dir = source_repo / "modified_file"
    target_modified_dir = target_repo / "modified_file"
    
    if not source_modified_dir.exists():
        print(f"  [警告] 源 modified_file 目录不存在: {source_modified_dir}")
        return success_count, len(commit_hashes), error_count
    
    for commit_hash in commit_hashes:
        source_commit_dir = source_modified_dir / commit_hash
        target_commit_dir = target_modified_dir / commit_hash
        
        if not source_commit_dir.exists():
            missing_count += 1
            continue
        
        try:
            # 如果目标已存在，先删除
            if target_commit_dir.exists():
                shutil.rmtree(target_commit_dir)
            
            # 复制整个目录
            shutil.copytree(source_commit_dir, target_commit_dir)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"  [错误] 复制失败 {repo_name}/{commit_hash}: {e}")
    
    return success_count, missing_count, error_count


def main():
    """主函数"""
    # 定义路径
    project_root = Path(__file__).parent
    all_commits_json = project_root / "huawei_stage2" / "all_is_opt_final.json"
    source_kb = project_root / "knowledge_base"
    target_kb = project_root / "huawei_stage2" / "knowledge_base"
    
    print("=" * 80)
    print("知识库数据提取和复制脚本")
    print("=" * 80)
    print(f"源 JSON 文件: {all_commits_json}")
    print(f"源知识库目录: {source_kb}")
    print(f"目标知识库目录: {target_kb}")
    print("=" * 80)
    
    # 检查输入文件和目录
    if not all_commits_json.exists():
        print(f"[错误] 找不到文件: {all_commits_json}")
        return
    
    if not source_kb.exists():
        print(f"[错误] 找不到源知识库目录: {source_kb}")
        return
    
    # 创建目标目录
    target_kb.mkdir(parents=True, exist_ok=True)
    
    # 第一步：加载并分组 commits
    commits = load_all_commits(str(all_commits_json))
    grouped_commits = group_commits_by_repository(commits)
    
    # 第二步：创建目标目录结构
    create_repository_directories(target_kb, list(grouped_commits.keys()))
    
    # 第三步和第四步：保存 JSON 并复制文件
    print("\n正在处理各代码库...")
    print("-" * 80)
    
    total_success = 0
    total_missing = 0
    total_error = 0
    
    for idx, (repo_name, repo_commits) in enumerate(grouped_commits.items(), 1):
        print(f"\n[{idx}/{len(grouped_commits)}] 处理代码库: {repo_name}")
        print(f"  Commit 数量: {len(repo_commits)}")
        
        # 保存 is_opt_final.json
        try:
            save_repository_json(target_kb, repo_name, repo_commits)
            print(f"  ✓ 已生成 is_opt_final.json")
        except Exception as e:
            print(f"  [错误] 保存 JSON 失败: {e}")
            continue
        
        # 收集所有 commit hash
        commit_hashes = {commit['hash'] for commit in repo_commits if 'hash' in commit}
        print(f"  需要复制的 commit 文件: {len(commit_hashes)}")
        
        # 复制 modified_file 目录
        success, missing, error = copy_commit_files(
            source_kb, target_kb, repo_name, commit_hashes
        )
        
        total_success += success
        total_missing += missing
        total_error += error
        
        if success > 0:
            print(f"  ✓ 成功复制: {success}")
        if missing > 0:
            print(f"  ⚠ 源文件缺失: {missing}")
        if error > 0:
            print(f"  ✗ 复制错误: {error}")
    
    # 输出统计摘要
    print("\n" + "=" * 80)
    print("处理完成！统计摘要:")
    print("=" * 80)
    print(f"代码库总数: {len(grouped_commits)}")
    print(f"Commit 总数: {len(commits)}")
    print(f"成功复制的 commit 文件夹: {total_success}")
    print(f"源文件缺失的 commit: {total_missing}")
    print(f"复制错误的 commit: {total_error}")
    print("=" * 80)
    
    # 验证结果
    print("\n验证目标目录结构...")
    created_repos = [d for d in target_kb.iterdir() if d.is_dir()]
    print(f"已创建 {len(created_repos)} 个代码库目录")
    
    json_files = list(target_kb.glob("*/is_opt_final.json"))
    print(f"已生成 {len(json_files)} 个 is_opt_final.json 文件")
    
    print("\n脚本执行完成！")


if __name__ == "__main__":
    main()

