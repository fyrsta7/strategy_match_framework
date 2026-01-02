#!/usr/bin/env python3
"""
统计 get_line_offset_new.py 脚本的处理进度

功能：
1. 统计总共需要处理的commit数量
2. 统计已完全处理的commit数量（包含所有4个字段）
3. 统计部分处理的commit数量（包含部分字段）
4. 统计未处理的commit数量
5. 统计无法处理的commit数量（缺少前置字段）
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

# get_line_offset_new.py 处理生成的字段
OUTPUT_FIELDS = ['line_offset', 'func_start_line', 'func_end_line', 'before_func_total_lines']

# get_line_offset_new.py 需要的前置字段
REQUIRED_FIELDS = ['file_start_line', 'file_end_line']


def analyze_commit_status(commit):
    """
    分析单个commit的处理状态
    返回: (status, details)
    
    status 可能的值:
    - 'complete': 所有4个字段都存在且非空
    - 'partial': 部分字段存在
    - 'unprocessed': 没有任何字段，但有前置字段，可以处理
    - 'missing_prerequisites': 缺少前置字段，无法处理
    """
    details = {
        'has_line_offset': False,
        'has_func_start_line': False,
        'has_func_end_line': False,
        'has_before_func_total_lines': False,
        'has_file_start_line': False,
        'has_file_end_line': False,
        'present_fields': [],
        'missing_fields': []
    }
    
    # 检查输出字段
    for field in OUTPUT_FIELDS:
        if field in commit and commit[field] is not None:
            details[f'has_{field}'] = True
            details['present_fields'].append(field)
        else:
            details['missing_fields'].append(field)
    
    # 检查前置字段
    for field in REQUIRED_FIELDS:
        if field in commit and commit[field] is not None:
            details[f'has_{field}'] = True
    
    # 判断状态
    present_count = len(details['present_fields'])
    
    if present_count == len(OUTPUT_FIELDS):
        return 'complete', details
    elif present_count > 0:
        return 'partial', details
    elif details['has_file_start_line'] and details['has_file_end_line']:
        return 'unprocessed', details
    else:
        return 'missing_prerequisites', details


def analyze_progress(repositories):
    """
    分析处理进度
    """
    print("\n[1/3] 分析处理进度...")
    
    stats = {
        'total_repos': 0,
        'repos_with_file': 0,
        'total_commits': 0,
        'complete_commits': 0,
        'partial_commits': 0,
        'unprocessed_commits': 0,
        'missing_prerequisites_commits': 0,
    }
    
    # 字段级别统计
    field_stats = {field: 0 for field in OUTPUT_FIELDS}
    
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
            'complete': 0,
            'partial': 0,
            'unprocessed': 0,
            'missing_prerequisites': 0,
            'field_counts': {field: 0 for field in OUTPUT_FIELDS}
        }
        
        for commit in commits:
            stats['total_commits'] += 1
            repo_stat['total'] += 1
            
            status, details = analyze_commit_status(commit)
            
            if status == 'complete':
                stats['complete_commits'] += 1
                repo_stat['complete'] += 1
                for field in OUTPUT_FIELDS:
                    field_stats[field] += 1
                    repo_stat['field_counts'][field] += 1
            elif status == 'partial':
                stats['partial_commits'] += 1
                repo_stat['partial'] += 1
                for field in details['present_fields']:
                    field_stats[field] += 1
                    repo_stat['field_counts'][field] += 1
            elif status == 'unprocessed':
                stats['unprocessed_commits'] += 1
                repo_stat['unprocessed'] += 1
            else:  # missing_prerequisites
                stats['missing_prerequisites_commits'] += 1
                repo_stat['missing_prerequisites'] += 1
        
        # 只记录有未完成任务的代码库
        if repo_stat['unprocessed'] > 0 or repo_stat['partial'] > 0 or repo_stat['missing_prerequisites'] > 0:
            repo_details.append(repo_stat)
    
    return stats, field_stats, repo_details


def print_statistics(stats, field_stats, repo_details):
    """
    打印统计信息
    """
    print("\n" + "=" * 80)
    print("get_line_offset_new.py 处理进度统计报告")
    print("=" * 80)
    
    print(f"\n【代码库统计】")
    print(f"  总代码库数: {stats['total_repos']}")
    print(f"  包含 {TARGET_FILE} 的代码库: {stats['repos_with_file']}")
    print(f"  有未完成任务的代码库: {len(repo_details)}")
    
    print(f"\n【Commit 总体统计】")
    print(f"  总 commit 数: {stats['total_commits']}")
    
    if stats['total_commits'] > 0:
        complete_pct = stats['complete_commits'] / stats['total_commits'] * 100
        partial_pct = stats['partial_commits'] / stats['total_commits'] * 100
        unprocessed_pct = stats['unprocessed_commits'] / stats['total_commits'] * 100
        missing_pct = stats['missing_prerequisites_commits'] / stats['total_commits'] * 100
        
        print(f"  完全处理: {stats['complete_commits']} ({complete_pct:.2f}%)")
        print(f"  部分处理: {stats['partial_commits']} ({partial_pct:.2f}%)")
        print(f"  未处理: {stats['unprocessed_commits']} ({unprocessed_pct:.2f}%)")
        print(f"  缺少前置字段: {stats['missing_prerequisites_commits']} ({missing_pct:.2f}%)")
        
        # 可处理的总数
        processable = stats['total_commits'] - stats['missing_prerequisites_commits']
        if processable > 0:
            print(f"\n【可处理的 commit 分析】（排除缺少前置字段的）")
            print(f"  可处理总数: {processable}")
            complete_rate = stats['complete_commits'] / processable * 100
            remaining = processable - stats['complete_commits']
            remaining_rate = remaining / processable * 100
            print(f"  已完全处理: {stats['complete_commits']} ({complete_rate:.2f}%)")
            print(f"  剩余待处理: {remaining} ({remaining_rate:.2f}%)")
            print(f"    - 部分处理: {stats['partial_commits']}")
            print(f"    - 完全未处理: {stats['unprocessed_commits']}")
    
    # print(f"\n【字段级别统计】")
    # for field in OUTPUT_FIELDS:
    #     if stats['total_commits'] > 0:
    #         field_pct = field_stats[field] / stats['total_commits'] * 100
    #         print(f"  {field}: {field_stats[field]}/{stats['total_commits']} ({field_pct:.2f}%)")
    #     else:
    #         print(f"  {field}: 0")
    
    # # 打印有未完成任务的代码库详情
    # if repo_details:
    #     print(f"\n【有未完成任务的代码库详情】（显示前20个，按待处理数量降序）")
    #     print(f"{'代码库名称':<50} {'总数':>8} {'完成':>8} {'部分':>8} {'待处理':>8} {'缺前置':>8}")
    #     print("-" * 110)
        
    #     # 按待处理数量排序（部分+未处理）
    #     repo_details_sorted = sorted(
    #         repo_details, 
    #         key=lambda x: x['partial'] + x['unprocessed'], 
    #         reverse=True
    #     )
        
    #     for i, repo in enumerate(repo_details_sorted[:20], 1):
    #         name = repo['repo_name']
    #         if len(name) > 48:
    #             name = name[:45] + "..."
            
    #         remaining = repo['partial'] + repo['unprocessed']
            
    #         print(f"{name:<50} {repo['total']:>8} {repo['complete']:>8} {repo['partial']:>8} "
    #               f"{repo['unprocessed']:>8} {repo['missing_prerequisites']:>8}")
        
    #     if len(repo_details_sorted) > 20:
    #         print(f"\n  ... 还有 {len(repo_details_sorted) - 20} 个代码库未显示")
    
    # 如果存在缺少前置字段的commit，给出提示
    if stats['missing_prerequisites_commits'] > 0:
        print(f"\n【注意事项】")
        print(f"  有 {stats['missing_prerequisites_commits']} 个 commit 缺少前置字段 (file_start_line, file_end_line)")
        print(f"  这些 commit 需要先运行 get_line_num.py 来生成前置字段")
        print(f"  提示：运行 'python3 get_line_num.py' 来生成缺失的前置字段")


def export_detailed_report(repo_details, output_file="get_line_offset_progress_report.json"):
    """
    导出详细报告到JSON文件
    """
    print(f"\n[2/3] 导出详细报告...")
    
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    
    # 按待处理数量排序
    repo_details_sorted = sorted(
        repo_details, 
        key=lambda x: x['partial'] + x['unprocessed'], 
        reverse=True
    )
    
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


def find_problematic_commits(repositories, max_examples=10):
    """
    查找问题commit的示例
    """
    print(f"\n[3/3] 查找问题 commit 示例...")
    
    partial_examples = []
    missing_prereq_examples = []
    
    for repo_name in repositories:
        json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, TARGET_FILE)
        
        if not os.path.exists(json_file_path):
            continue
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                commits = json.load(f)
        except:
            continue
        
        for commit in commits:
            status, details = analyze_commit_status(commit)
            
            if status == 'partial' and len(partial_examples) < max_examples:
                partial_examples.append({
                    'repo': repo_name,
                    'hash': commit.get('hash', 'unknown'),
                    'present_fields': details['present_fields'],
                    'missing_fields': details['missing_fields']
                })
            
            if status == 'missing_prerequisites' and len(missing_prereq_examples) < max_examples:
                missing_prereq_examples.append({
                    'repo': repo_name,
                    'hash': commit.get('hash', 'unknown'),
                    'has_file_start_line': details['has_file_start_line'],
                    'has_file_end_line': details['has_file_end_line']
                })
            
            if len(partial_examples) >= max_examples and len(missing_prereq_examples) >= max_examples:
                break
        
        if len(partial_examples) >= max_examples and len(missing_prereq_examples) >= max_examples:
            break
    
    if partial_examples:
        print(f"\n【部分处理的 commit 示例】（前 {len(partial_examples)} 个）")
        for i, example in enumerate(partial_examples, 1):
            print(f"  {i}. {example['repo']} - {example['hash'][:12]}")
            print(f"     已有字段: {', '.join(example['present_fields'])}")
            print(f"     缺失字段: {', '.join(example['missing_fields'])}")
    
    if missing_prereq_examples:
        print(f"\n【缺少前置字段的 commit 示例】（前 {len(missing_prereq_examples)} 个）")
        for i, example in enumerate(missing_prereq_examples, 1):
            print(f"  {i}. {example['repo']} - {example['hash'][:12]}")
            print(f"     file_start_line: {'✓' if example['has_file_start_line'] else '✗'}")
            print(f"     file_end_line: {'✓' if example['has_file_end_line'] else '✗'}")


def main():
    """主函数"""
    print("=" * 80)
    print("get_line_offset_new.py 进度统计工具")
    print("=" * 80)
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"知识库路径: {KNOWLEDGE_BASE_PATH}")
    print(f"目标文件: {TARGET_FILE}")
    print(f"检查字段: {', '.join(OUTPUT_FIELDS)}")
    print(f"前置字段: {', '.join(REQUIRED_FIELDS)}")
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
    
    # 分析进度
    stats, field_stats, repo_details = analyze_progress(repositories)
    
    # 打印统计信息
    print_statistics(stats, field_stats, repo_details)
    
    # 导出详细报告
    # export_detailed_report(repo_details)
    
    # 查找问题commit示例
    # find_problematic_commits(repositories, max_examples=5)
    
    # print("\n" + "=" * 80)
    # print("统计完成！")
    # print("=" * 80)


if __name__ == "__main__":
    main()

