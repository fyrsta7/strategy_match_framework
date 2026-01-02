#!/usr/bin/env python3
"""
统计 summary.py 脚本的处理进度

功能：
1. 统计总共需要处理的commit数量
2. 统计已处理的commit数量
3. 统计剩余需要处理的commit数量
4. 分析剩余commit中有多少可以复用，多少需要重新生成
"""

import os
import json
import sys
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")
TARGET_FILE = "summary_huawei.json"

# 复用来源配置（与 summary.py 保持一致）
REUSE_SOURCES = [
    {
        'summary_file': 'summary.json',
        'field_name': 'optimization_summary_final',
        'label': 'general'
    },
    {
        'summary_file': 'summary_arch.json',
        'field_name': 'optimization_summary_arch_final',
        'label': 'arch'
    }
]


def build_reuse_index(repositories):
    """
    构建复用索引：{(repo_name, commit_hash): (summary, source_label)}
    """
    print("\n[1/4] 构建复用索引...")
    index = {}
    stats = {
        'general_found': 0,
        'arch_found': 0,
        'total_indexed': 0
    }
    
    for repo_name in tqdm(repositories, desc="构建索引", unit="repo"):
        for source_config in REUSE_SOURCES:
            file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, source_config['summary_file'])
            
            if not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    commits = json.load(f)
                
                for commit in commits:
                    commit_repo = commit.get('repository_name', repo_name)
                    commit_hash = commit.get('hash')
                    
                    if not commit_hash:
                        continue
                    
                    key = (commit_repo, commit_hash)
                    
                    # 优先使用第一个找到的 summary
                    if key not in index:
                        summary = commit.get(source_config['field_name'])
                        if summary and summary.strip():
                            index[key] = (summary, source_config['label'])
                            stats['total_indexed'] += 1
                            
                            if source_config['label'] == 'general':
                                stats['general_found'] += 1
                            elif source_config['label'] == 'arch':
                                stats['arch_found'] += 1
            
            except Exception as e:
                pass
    
    print(f"  从 summary.json 索引: {stats['general_found']} 个")
    print(f"  从 summary_arch.json 索引: {stats['arch_found']} 个")
    print(f"  总索引数: {stats['total_indexed']} 个")
    
    return index


def analyze_progress(repositories, reuse_index):
    """
    分析处理进度
    """
    print("\n[2/4] 分析处理进度...")
    
    stats = {
        'total_repos': 0,
        'repos_with_file': 0,
        'total_commits': 0,
        'processed_commits': 0,
        'remaining_commits': 0,
        'remaining_can_reuse': 0,
        'remaining_need_generate': 0,
        'reused_from_general': 0,
        'reused_from_arch': 0,
        'generated': 0,
    }
    
    # 详细统计
    repo_details = []
    
    for repo_name in tqdm(repositories, desc="分析进度", unit="repo"):
        stats['total_repos'] += 1
        
        json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, TARGET_FILE)
        
        if not os.path.exists(json_file_path):
            continue
        
        stats['repos_with_file'] += 1
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                commits = json.load(f)
        except Exception as e:
            print(f"\n[错误] 读取文件失败 {repo_name}: {e}")
            continue
        
        if not commits:
            continue
        
        repo_stat = {
            'repo_name': repo_name,
            'total': len(commits),
            'processed': 0,
            'remaining': 0,
            'remaining_reusable': 0,
            'remaining_need_generate': 0
        }
        
        for commit in commits:
            stats['total_commits'] += 1
            repo_stat['total'] += 1
            
            # 检查是否已处理
            if commit.get('optimization_summary_huawei_final'):
                stats['processed_commits'] += 1
                repo_stat['processed'] += 1
                
                # 统计已处理的来源
                reused_from = commit.get('reused_from', 'unknown')
                if reused_from == 'general':
                    stats['reused_from_general'] += 1
                elif reused_from == 'arch':
                    stats['reused_from_arch'] += 1
                elif reused_from == 'generated':
                    stats['generated'] += 1
            else:
                # 未处理的commit
                stats['remaining_commits'] += 1
                repo_stat['remaining'] += 1
                
                # 检查是否可以复用
                commit_repo = commit.get('repository_name', repo_name)
                commit_hash = commit.get('hash')
                key = (commit_repo, commit_hash)
                
                if key in reuse_index:
                    stats['remaining_can_reuse'] += 1
                    repo_stat['remaining_reusable'] += 1
                else:
                    stats['remaining_need_generate'] += 1
                    repo_stat['remaining_need_generate'] += 1
        
        # 只记录有剩余任务的代码库
        if repo_stat['remaining'] > 0:
            repo_details.append(repo_stat)
    
    return stats, repo_details


def print_statistics(stats, repo_details):
    """
    打印统计信息
    """
    print("\n" + "=" * 80)
    print("处理进度统计报告")
    print("=" * 80)
    
    print(f"\n【代码库统计】")
    print(f"  总代码库数: {stats['total_repos']}")
    print(f"  包含 {TARGET_FILE} 的代码库: {stats['repos_with_file']}")
    print(f"  有剩余任务的代码库: {len(repo_details)}")
    
    print(f"\n【Commit 总体统计】")
    print(f"  总 commit 数: {stats['total_commits']}")
    print(f"  已处理: {stats['processed_commits']} ({stats['processed_commits']/stats['total_commits']*100:.2f}%)")
    print(f"  剩余未处理: {stats['remaining_commits']} ({stats['remaining_commits']/stats['total_commits']*100:.2f}%)")
    
    if stats['processed_commits'] > 0:
        print(f"\n【已处理 commit 来源分析】")
        print(f"  从 summary.json 复用: {stats['reused_from_general']} ({stats['reused_from_general']/stats['processed_commits']*100:.2f}%)")
        print(f"  从 summary_arch.json 复用: {stats['reused_from_arch']} ({stats['reused_from_arch']/stats['processed_commits']*100:.2f}%)")
        print(f"  新生成: {stats['generated']} ({stats['generated']/stats['processed_commits']*100:.2f}%)")
    
    if stats['remaining_commits'] > 0:
        print(f"\n【剩余 commit 分析】")
        print(f"  可以复用: {stats['remaining_can_reuse']} ({stats['remaining_can_reuse']/stats['remaining_commits']*100:.2f}%)")
        print(f"  需要生成: {stats['remaining_need_generate']} ({stats['remaining_need_generate']/stats['remaining_commits']*100:.2f}%)")
        
        print(f"\n【预估工作量】")
        # 假设每个需要生成的commit调用3次LLM（NUM_SUMMARIES=3）
        estimated_llm_calls = stats['remaining_need_generate'] * 3
        print(f"  预估需要 LLM 调用: {stats['remaining_need_generate']} × 3 = {estimated_llm_calls} 次")
        print(f"  可节省 LLM 调用: {stats['remaining_can_reuse']} × 3 = {stats['remaining_can_reuse'] * 3} 次")
    
    # 打印有剩余任务的代码库详情（前20个）
    # if repo_details:
    #     print(f"\n【有剩余任务的代码库详情】（显示前20个，按剩余数量降序）")
    #     print(f"{'代码库名称':<50} {'总数':>8} {'已处理':>8} {'剩余':>8} {'可复用':>8} {'需生成':>8}")
    #     print("-" * 110)
        
    #     # 按剩余数量排序
    #     repo_details_sorted = sorted(repo_details, key=lambda x: x['remaining'], reverse=True)
        
    #     for i, repo in enumerate(repo_details_sorted[:20], 1):
    #         name = repo['repo_name']
    #         if len(name) > 48:
    #             name = name[:45] + "..."
            
    #         print(f"{name:<50} {repo['total']:>8} {repo['processed']:>8} {repo['remaining']:>8} "
    #               f"{repo['remaining_reusable']:>8} {repo['remaining_need_generate']:>8}")
        
    #     if len(repo_details_sorted) > 20:
    #         print(f"\n  ... 还有 {len(repo_details_sorted) - 20} 个代码库未显示")


def export_detailed_report(repo_details, output_file="progress_report.json"):
    """
    导出详细报告到JSON文件
    """
    print(f"\n[3/4] 导出详细报告...")
    
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    
    # 按剩余数量排序
    repo_details_sorted = sorted(repo_details, key=lambda x: x['remaining'], reverse=True)
    
    report = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'total_repos_with_remaining': len(repo_details_sorted),
        'repos': repo_details_sorted
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  详细报告已保存到: {output_path}")
    except Exception as e:
        print(f"  [错误] 保存报告失败: {e}")


def main():
    """主函数"""
    print("=" * 80)
    print("summary.py 进度统计工具")
    print("=" * 80)
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"知识库路径: {KNOWLEDGE_BASE_PATH}")
    print(f"目标文件: {TARGET_FILE}")
    print("-" * 80)
    
    # 读取代码库列表
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件不存在 - {REPO_LIST_FILE}")
        return
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repo_list = json.load(f)
    
    repositories = []
    for repo in repo_list:
        repo_name = repo.get('name_long') or repo.get('name')
        if repo_name:
            repositories.append(repo_name)
    
    if not repositories:
        print("错误：未找到任何代码库")
        return
    
    print(f"总共 {len(repositories)} 个代码库")
    
    # 构建复用索引
    reuse_index = build_reuse_index(repositories)
    
    # 分析进度
    stats, repo_details = analyze_progress(repositories, reuse_index)
    
    # 打印统计信息
    print("\n[4/4] 生成统计报告...")
    print_statistics(stats, repo_details)
    
    # 导出详细报告
    # export_detailed_report(repo_details)
    
    # print("\n" + "=" * 80)
    # print("统计完成！")
    # print("=" * 80)


if __name__ == "__main__":
    main()

