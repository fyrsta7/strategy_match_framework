#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm

# ========== 配置变量 ==========
# 知识库根目录
KNOWLEDGE_BASE_ROOT = os.path.join("/home/zyw/llm_on_code/llm_on_code_optimization/", "knowledge_base")

# 目标文件夹名称
TARGET_FOLDER_NAME = "semgrep"

# 是否需要用户确认（设为False自动执行）
REQUIRE_CONFIRMATION = True

# 是否试运行模式（True时只扫描不删除）
DRY_RUN_MODE = False


def find_semgrep_folders():
    """
    扫描knowledge_base目录，在commit文件夹中找到所有semgrep文件夹
    路径结构: knowledge_base/<repo_name>/modified_file/<commit_hash>/semgrep/
    返回: semgrep文件夹路径列表
    """
    semgrep_folders = []
    
    if not os.path.exists(KNOWLEDGE_BASE_ROOT):
        print(f"错误: 知识库目录不存在 - {KNOWLEDGE_BASE_ROOT}")
        return semgrep_folders
    
    print(f"正在扫描目录: {KNOWLEDGE_BASE_ROOT}")
    print("查找路径结构: <repo_name>/modified_file/<commit_hash>/semgrep/")
    
    # 遍历 knowledge_base/<repo_name>/
    for repo_name in os.listdir(KNOWLEDGE_BASE_ROOT):
        repo_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name)
        
        # 跳过非目录文件
        if not os.path.isdir(repo_path):
            continue
        
        print(f"正在扫描仓库: {repo_name}")
        
        # 检查是否存在 modified_file 文件夹
        modified_file_path = os.path.join(repo_path, "modified_file")
        if not os.path.exists(modified_file_path) or not os.path.isdir(modified_file_path):
            print(f"  跳过 {repo_name}: 没有找到 modified_file 文件夹")
            continue
        
        # 遍历 knowledge_base/<repo_name>/modified_file/<commit_hash>/
        for commit_hash in os.listdir(modified_file_path):
            commit_path = os.path.join(modified_file_path, commit_hash)
            
            # 跳过非目录文件
            if not os.path.isdir(commit_path):
                continue
            
            # 检查是否存在semgrep文件夹
            semgrep_path = os.path.join(commit_path, TARGET_FOLDER_NAME)
            if os.path.exists(semgrep_path) and os.path.isdir(semgrep_path):
                semgrep_folders.append(semgrep_path)
                print(f"找到: {semgrep_path}")
    
    return semgrep_folders


def get_folder_size(folder_path):
    """
    计算文件夹大小（MB）
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"警告: 无法计算文件夹大小 {folder_path}: {e}")
    
    return total_size / (1024 * 1024)  # 转换为MB


def delete_semgrep_folders(semgrep_folders):
    """
    删除semgrep文件夹并返回统计结果
    """
    stats = {
        "total": len(semgrep_folders),
        "success": 0,
        "failed": 0,
        "total_size_mb": 0,
        "errors": []
    }
    
    if not semgrep_folders:
        print("没有找到需要删除的semgrep文件夹")
        return stats
    
    print(f"\n{'='*50}")
    print(f"准备删除 {len(semgrep_folders)} 个semgrep文件夹:")
    
    # 显示将要删除的文件夹和大小
    total_size_mb = 0
    for folder_path in semgrep_folders:
        size_mb = get_folder_size(folder_path)
        total_size_mb += size_mb
        print(f"  - {folder_path} ({size_mb:.2f} MB)")
    
    print(f"总大小: {total_size_mb:.2f} MB")
    stats["total_size_mb"] = total_size_mb
    
    # 用户确认
    if REQUIRE_CONFIRMATION:
        if DRY_RUN_MODE:
            print(f"\n[试运行模式] 以上文件夹将会被删除")
            return stats
        else:
            confirm = input(f"\n确认删除以上 {len(semgrep_folders)} 个文件夹吗? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("操作已取消")
                return stats
    
    # 执行删除
    print(f"\n开始删除semgrep文件夹...")
    
    with tqdm(semgrep_folders, desc="删除进度", unit="folder") as pbar:
        for folder_path in pbar:
            pbar.set_postfix({"当前": os.path.basename(os.path.dirname(folder_path))})
            
            try:
                # 安全检查：确保路径在knowledge_base目录内
                if not folder_path.startswith(KNOWLEDGE_BASE_ROOT):
                    raise ValueError(f"路径不在安全范围内: {folder_path}")
                
                # 确保是semgrep文件夹
                if not folder_path.endswith(TARGET_FOLDER_NAME):
                    raise ValueError(f"路径不是{TARGET_FOLDER_NAME}文件夹: {folder_path}")
                
                # 执行删除
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    stats["success"] += 1
                    pbar.set_postfix({"状态": "删除成功"})
                else:
                    print(f"\n警告: 文件夹不存在 {folder_path}")
                    
            except Exception as e:
                stats["failed"] += 1
                error_msg = f"删除失败 {folder_path}: {str(e)}"
                stats["errors"].append(error_msg)
                print(f"\n错误: {error_msg}")
                pbar.set_postfix({"状态": "删除失败"})
    
    return stats


def print_statistics(stats):
    """
    打印统计结果
    """
    print(f"\n{'='*50}")
    print(f"删除操作统计结果:")
    print(f"{'='*50}")
    print(f"总计文件夹数: {stats['total']}")
    print(f"删除成功: {stats['success']}")
    print(f"删除失败: {stats['failed']}")
    print(f"释放空间: {stats['total_size_mb']:.2f} MB")
    
    if stats['errors']:
        print(f"\n错误详情:")
        for error in stats['errors']:
            print(f"  - {error}")
    
    if stats['success'] > 0:
        print(f"\n✅ 成功删除了 {stats['success']} 个semgrep文件夹")
    
    if stats['failed'] > 0:
        print(f"\n❌ {stats['failed']} 个文件夹删除失败")
    
    print(f"{'='*50}")


def main():
    """
    主函数
    """
    print("Semgrep文件夹清理工具")
    print(f"目标目录: {KNOWLEDGE_BASE_ROOT}")
    print(f"目标文件夹: {TARGET_FOLDER_NAME}")
    print(f"试运行模式: {'开启' if DRY_RUN_MODE else '关闭'}")
    print(f"需要确认: {'是' if REQUIRE_CONFIRMATION else '否'}")
    
    # 检查知识库目录
    if not os.path.exists(KNOWLEDGE_BASE_ROOT):
        print(f"错误: 知识库目录不存在 - {KNOWLEDGE_BASE_ROOT}")
        return
    
    # 扫描semgrep文件夹
    semgrep_folders = find_semgrep_folders()
    
    if not semgrep_folders:
        print("未找到任何semgrep文件夹，无需清理。")
        return
    
    # 执行删除操作
    stats = delete_semgrep_folders(semgrep_folders)
    
    # 打印统计结果
    print_statistics(stats)


if __name__ == "__main__":
    main()