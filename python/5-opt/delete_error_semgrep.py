"""
清理错误的 Semgrep 规则文件

功能：
1. 扫描所有聚类中的 commit
2. 识别已生成规则但JSON结果包含fatal/error错误的规则（needs_fixing状态）
3. 删除这些错误的规则文件（YAML 和 JSON）
4. 生成清理报告

注意：
- 只删除已生成规则但运行结果有错误的规则
- 不包括未生成的规则（both_missing, yaml_missing, json_missing）
- 不包括JSON解析错误（json_parse_error）

使用方法：
    python delete_error_semgrep.py
"""

import os
import json
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from import_configs import global_config, opt_config

def print_info(message):
    """关键信息输出 - 始终显示"""
    print(message, flush=True)

def print_verbose(message):
    """详细信息输出 - 仅在VERBOSE模式下显示"""
    if opt_config.VERBOSE:
        print(message, flush=True)

def print_progress(message):
    """进度信息输出 - 始终显示，用于追踪运行进展"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def find_before_file(commit_dir):
    """查找before文件（任意后缀）"""
    pattern = os.path.join(commit_dir, "before.*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None

def build_commit_paths(commit_info):
    """构建commit相关的文件路径"""
    knowledge_base_path = os.path.join(global_config.root_path, opt_config.KNOWLEDGE_BASE_RELATIVE_PATH)
    repo_name = commit_info['repository_name']
    commit_hash = commit_info['hash']
    commit_dir = os.path.join(knowledge_base_path, repo_name, 'modified_file', commit_hash)
    
    before_file_path = find_before_file(commit_dir)
    return {
        'commit_dir': commit_dir,
        'diff_file': os.path.join(commit_dir, 'diff.txt'),
        'before_file': before_file_path,
        'semgrep_dir': os.path.join(commit_dir, 'semgrep')
    }

def is_commit_processable(commit_info):
    """检查commit是否可以处理"""
    required_fields = ['repository_name', 'hash', 'file_start_line', 'file_end_line']
    for field in required_fields:
        if field not in commit_info:
            return False, f"Missing required field: {field}"
    
    try:
        paths = build_commit_paths(commit_info)
        if not os.path.exists(paths['diff_file']):
            return False, "diff.txt not found"
        if not paths['before_file'] or not os.path.exists(paths['before_file']):
            return False, "before file not found"
        
        with open(paths['diff_file'], 'r', encoding='utf-8') as f:
            if not f.read().strip():
                return False, "diff.txt is empty"
        
        return True, None
    except Exception as e:
        return False, f"Error checking files: {str(e)}"

def should_continue_fixing(json_result, error_msg):
    """判断是否需要继续修复迭代（复用process_once.py的逻辑）"""
    if error_msg:
        return True
    
    if not json_result or not json_result.get("errors"):
        return False
    
    for error in json_result.get("errors", []):
        level = error.get("level", "").lower()
        if level in ["fatal", "error"]:
            return True
    
    return False

def analyze_single_rule(rule_number, semgrep_dir):
    """分析单个规则的状态"""
    yaml_path = os.path.join(semgrep_dir, f"{rule_number}.yaml")
    json_path = os.path.join(semgrep_dir, f"{rule_number}.json")
    
    yaml_exists = os.path.exists(yaml_path) and os.path.getsize(yaml_path) > 0
    json_exists = os.path.exists(json_path) and os.path.getsize(json_path) > 0
    
    rule_analysis = {
        "rule_number": rule_number,
        "yaml_exists": yaml_exists,
        "json_exists": json_exists,
        "status": "unknown",
        "error_message": None,
        "error_details": []
    }
    
    if not yaml_exists and not json_exists:
        rule_analysis["status"] = "both_missing"
        rule_analysis["error_message"] = "Both YAML and JSON files missing"
        return rule_analysis
    elif not yaml_exists:
        rule_analysis["status"] = "yaml_missing"
        rule_analysis["error_message"] = "YAML file missing"
        return rule_analysis
    elif not json_exists:
        rule_analysis["status"] = "json_missing"
        rule_analysis["error_message"] = "JSON result file missing"
        return rule_analysis
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_result = json.load(f)
        
        needs_fixing = should_continue_fixing(json_result, None)
        
        if needs_fixing:
            rule_analysis["status"] = "needs_fixing"
            if json_result.get("errors"):
                for error in json_result["errors"]:
                    level = error.get("level", "unknown")
                    if level.lower() in ["fatal", "error"]:
                        rule_analysis["error_details"].append({
                            "level": level,
                            "message": error.get("message", "No message"),
                            "type": error.get("type", "Unknown")
                        })
                rule_analysis["error_message"] = f"Contains {len(rule_analysis['error_details'])} fatal/error level issues"
            else:
                rule_analysis["error_message"] = "Unknown error condition"
        else:
            rule_analysis["status"] = "usable"
            warning_count = 0
            if json_result.get("errors"):
                for error in json_result["errors"]:
                    level = error.get("level", "unknown").lower()
                    if level in ["warning", "info"]:
                        warning_count += 1
            
            if warning_count > 0:
                rule_analysis["error_message"] = f"Usable with {warning_count} warning(s)"
            else:
                rule_analysis["error_message"] = "No errors"
                
    except Exception as e:
        rule_analysis["status"] = "json_parse_error"
        rule_analysis["error_message"] = f"Error parsing JSON: {str(e)}"
    
    return rule_analysis

def is_error_rule(rule_analysis):
    """判断规则是否为错误规则（需要删除）
    
    只考虑已经生成了规则文件，但JSON结果中包含fatal/error错误的规则。
    不包括：
    - both_missing: 两个文件都不存在（规则还没生成）
    - yaml_missing: YAML文件缺失（规则还没生成）
    - json_missing: JSON文件缺失（规则还没运行）
    - json_parse_error: JSON解析错误（可能是文件损坏，但规则可能没问题）
    
    只包括：
    - needs_fixing: YAML和JSON都存在，但JSON中包含fatal/error级别错误
    """
    # 只处理 needs_fixing 状态，并且确保YAML和JSON都存在
    if rule_analysis["status"] == "needs_fixing":
        # 额外检查：确保两个文件都存在（needs_fixing状态应该已经保证了这一点）
        if rule_analysis.get("yaml_exists") and rule_analysis.get("json_exists"):
            return True
    return False

def cleanup_single_commit(commit_info, cluster_id, commit_idx, total_commits_in_cluster):
    """清理单个commit的错误规则"""
    commit_hash = commit_info['hash']
    repo_name = commit_info['repository_name']
    
    cleanup_result = {
        "repository_name": repo_name,
        "hash": commit_hash,
        "cluster_id": cluster_id,
        "rules_checked": 0,
        "error_rules_found": 0,
        "rules_deleted": 0,
        "deleted_rule_numbers": [],
        "error_details": []
    }
    
    try:
        paths = build_commit_paths(commit_info)
        semgrep_dir = paths['semgrep_dir']
        
        if not os.path.exists(semgrep_dir):
            return True, cleanup_result
        
        print_verbose(f"Checking commit {repo_name}:{commit_hash[:8]} ({commit_idx}/{total_commits_in_cluster})")
        
        # 检查所有可能的规则文件（1 到 COMMIT_GENERATION_COUNT）
        for rule_number in range(1, opt_config.COMMIT_GENERATION_COUNT + 1):
            cleanup_result["rules_checked"] += 1
            rule_analysis = analyze_single_rule(rule_number, semgrep_dir)
            
            if is_error_rule(rule_analysis):
                cleanup_result["error_rules_found"] += 1
                
                # 删除 YAML 和 JSON 文件
                yaml_path = os.path.join(semgrep_dir, f"{rule_number}.yaml")
                json_path = os.path.join(semgrep_dir, f"{rule_number}.json")
                
                deleted_files = []
                if os.path.exists(yaml_path):
                    try:
                        os.remove(yaml_path)
                        deleted_files.append("yaml")
                    except Exception as e:
                        cleanup_result["error_details"].append({
                            "rule_number": rule_number,
                            "error": f"Failed to delete YAML: {str(e)}"
                        })
                
                if os.path.exists(json_path):
                    try:
                        os.remove(json_path)
                        deleted_files.append("json")
                    except Exception as e:
                        cleanup_result["error_details"].append({
                            "rule_number": rule_number,
                            "error": f"Failed to delete JSON: {str(e)}"
                        })
                
                if deleted_files:
                    cleanup_result["rules_deleted"] += 1
                    cleanup_result["deleted_rule_numbers"].append(rule_number)
                    print_verbose(f"  Deleted rule {rule_number} ({rule_analysis['status']}): {', '.join(deleted_files)}")
        
        return True, cleanup_result
        
    except Exception as e:
        cleanup_result["error_message"] = str(e)
        return False, cleanup_result

def cleanup_single_cluster(cluster_info, cluster_index, total_clusters):
    """清理单个聚类中所有commit的错误规则"""
    cluster_id = cluster_info.get('cluster_id', f"cluster_{cluster_index}")
    
    cluster_cleanup_result = {
        "cluster_id": cluster_id,
        "cluster_index": cluster_index,
        "commits_checked": 0,
        "commits_with_errors": 0,
        "total_rules_checked": 0,
        "total_error_rules_found": 0,
        "total_rules_deleted": 0,
        "commit_details": []
    }
    
    print_progress(f"Cleaning cluster {cluster_id} ({cluster_index + 1}/{total_clusters})")
    
    # 筛选可处理的commits
    processable_commits = []
    for commit in cluster_info['commits']:
        processable, _ = is_commit_processable(commit)
        if processable:
            processable_commits.append(commit)
    
    if not processable_commits:
        return cluster_cleanup_result
    
    # 只处理前 SIMPLE_COMMITS_PER_CLUSTER 个 commit
    commits_to_check = processable_commits[:opt_config.SIMPLE_COMMITS_PER_CLUSTER]
    total_commits_in_cluster = len(commits_to_check)
    
    # 并行处理commits
    with ThreadPoolExecutor(max_workers=opt_config.SIMPLE_COMMIT_MAX_WORKERS) as executor:
        future_to_commit = {}
        for commit_idx, commit in enumerate(commits_to_check, 1):
            future = executor.submit(cleanup_single_commit, commit, cluster_id, commit_idx, total_commits_in_cluster)
            future_to_commit[future] = commit
        
        for future in as_completed(future_to_commit):
            success, commit_result = future.result()
            
            cluster_cleanup_result["commits_checked"] += 1
            cluster_cleanup_result["total_rules_checked"] += commit_result["rules_checked"]
            
            if commit_result["error_rules_found"] > 0:
                cluster_cleanup_result["commits_with_errors"] += 1
                cluster_cleanup_result["total_error_rules_found"] += commit_result["error_rules_found"]
                cluster_cleanup_result["total_rules_deleted"] += commit_result["rules_deleted"]
            
            cluster_cleanup_result["commit_details"].append(commit_result)
    
    print_progress(f"Cluster {cluster_id} completed: {cluster_cleanup_result['commits_with_errors']} commits with errors, "
                  f"{cluster_cleanup_result['total_rules_deleted']} rules deleted")
    
    return cluster_cleanup_result

def cleanup_clusters_parallel(clusters, max_workers=None):
    """并行清理多个聚类中的错误规则"""
    if max_workers is None:
        max_workers = opt_config.SIMPLE_CLUSTER_MAX_WORKERS
    
    results = []
    clusters_to_process = clusters
    if opt_config.SIMPLE_PROCESS_CLUSTER_LIMIT > 0:
        clusters_to_process = clusters[:opt_config.SIMPLE_PROCESS_CLUSTER_LIMIT]
    
    if not clusters_to_process:
        return results
    
    total_clusters = len(clusters_to_process)
    print_info(f"开始并行清理 {total_clusters} 个聚类中的错误规则，使用 {max_workers} 个并行工作线程")
    
    tqdm_kwargs = {
        'desc': "清理聚类中",
        'unit': "个聚类",
        'total': total_clusters,
        'dynamic_ncols': True,
        'position': 0,
        'leave': True
    }
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cluster = {}
        for i, cluster in enumerate(clusters_to_process):
            future = executor.submit(cleanup_single_cluster, cluster, i, total_clusters)
            future_to_cluster[future] = i
        
        print_info("所有清理任务已提交，等待完成...")
        
        with tqdm(**tqdm_kwargs) as pbar:
            for future in as_completed(future_to_cluster):
                cluster_idx = future_to_cluster[future]
                try:
                    cluster_result = future.result()
                    results.append(cluster_result)
                except Exception as e:
                    error_result = {
                        "cluster_id": f"cluster_{cluster_idx}",
                        "cluster_index": cluster_idx,
                        "error_message": f"Cluster cleanup failed: {str(e)}"
                    }
                    results.append(error_result)
                    print_progress(f"Cluster {cluster_idx} failed with exception: {str(e)}")
                finally:
                    pbar.update(1)
                    pbar.refresh()
    
    results.sort(key=lambda x: x['cluster_index'])
    print_info("所有聚类清理完成")
    return results

def generate_cleanup_statistics(cluster_results):
    """生成清理统计信息"""
    total_clusters = len(cluster_results)
    valid_clusters = [r for r in cluster_results if 'error_message' not in r]
    
    total_commits_checked = sum(r.get('commits_checked', 0) for r in valid_clusters)
    total_commits_with_errors = sum(r.get('commits_with_errors', 0) for r in valid_clusters)
    total_rules_checked = sum(r.get('total_rules_checked', 0) for r in valid_clusters)
    total_error_rules_found = sum(r.get('total_error_rules_found', 0) for r in valid_clusters)
    total_rules_deleted = sum(r.get('total_rules_deleted', 0) for r in valid_clusters)
    
    return {
        "total_clusters": total_clusters,
        "valid_clusters": len(valid_clusters),
        "failed_clusters": total_clusters - len(valid_clusters),
        "total_commits_checked": total_commits_checked,
        "total_commits_with_errors": total_commits_with_errors,
        "total_rules_checked": total_rules_checked,
        "total_error_rules_found": total_error_rules_found,
        "total_rules_deleted": total_rules_deleted,
        "error_rate": total_error_rules_found / total_rules_checked if total_rules_checked > 0 else 0
    }

def print_cleanup_statistics(overall_statistics, duration_seconds):
    """打印清理统计信息（中文）"""
    print_info(f"\n{'='*80}")
    print_info(f"错误规则清理报告")
    print_info(f"{'='*80}")
    
    print_info(f"【整体统计】")
    print_info(f"  聚类总数: {overall_statistics['total_clusters']}")
    print_info(f"  有效聚类: {overall_statistics['valid_clusters']}")
    print_info(f"  失败聚类: {overall_statistics['failed_clusters']}")
    print_info(f"")
    
    print_info(f"【Commit 统计】")
    print_info(f"  检查的 Commit 数: {overall_statistics['total_commits_checked']}")
    print_info(f"  有错误规则的 Commit 数: {overall_statistics['total_commits_with_errors']}")
    print_info(f"")
    
    print_info(f"【规则统计】")
    print_info(f"  检查的规则总数: {overall_statistics['total_rules_checked']}")
    print_info(f"  发现的错误规则数: {overall_statistics['total_error_rules_found']}")
    print_info(f"  删除的规则数: {overall_statistics['total_rules_deleted']}")
    print_info(f"  错误率: {overall_statistics['error_rate']:.1%}")
    print_info(f"")
    
    print_info(f"  总耗时: {duration_seconds} 秒")
    print_info(f"{'='*80}")

def save_cleanup_result(result_data, output_path):
    """保存清理结果到JSON文件"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=opt_config.DEFAULT_OUTPUT_INDENT, ensure_ascii=False)
        print_info(f"清理结果已保存到: {output_path}")
        return True
    except Exception as e:
        print_info(f"保存清理结果时出错: {e}")
        return False

def cleanup_cluster_file(cluster_json_path):
    """清理聚类JSON文件中所有commit的错误规则"""
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    
    print_info(f"清理开始于 {start_time_str}")
    
    if not os.path.exists(cluster_json_path):
        print_info(f"错误: 未找到聚类 JSON 文件: {cluster_json_path}")
        return None
    
    try:
        with open(cluster_json_path, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
    except Exception as e:
        print_info(f"读取聚类 JSON 文件时出错: {e}")
        return None
    
    if 'clusters' not in cluster_data:
        print_info(f"错误: JSON 文件中未找到 'clusters' 字段")
        return None
    
    clusters = cluster_data['clusters']
    print_info(f"从输入文件加载了 {len(clusters)} 个聚类")
    
    # 并行清理聚类
    cluster_results = cleanup_clusters_parallel(clusters)
    
    # 记录结束时间和计算耗时
    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    duration_seconds = int(end_time - start_time)
    print_info(f"清理完成于 {end_time_str} (耗时: {duration_seconds} 秒)")
    
    # 生成整体统计
    overall_statistics = generate_cleanup_statistics(cluster_results)
    
    # 输出详细统计信息
    print_cleanup_statistics(overall_statistics, duration_seconds)
    
    return {
        "overall_statistics": overall_statistics,
        "cluster_results": cluster_results
    }

def main():
    """主函数"""
    CLUSTER_JSON_PATH = global_config.root_path + "python/2-cluster/result_1000/0_8_2_order.json"
    
    print_info(f"{'='*80}")
    print_info(f"错误 Semgrep 规则清理工具")
    print_info(f"{'='*80}")
    print_verbose(f"聚类 JSON 文件: {CLUSTER_JSON_PATH}")
    print_verbose(f"知识库路径: {os.path.join(global_config.root_path, opt_config.KNOWLEDGE_BASE_RELATIVE_PATH)}")
    print_info(f"处理聚类数量限制: {opt_config.SIMPLE_PROCESS_CLUSTER_LIMIT if opt_config.SIMPLE_PROCESS_CLUSTER_LIMIT > 0 else '全部'}")
    print_info(f"每个聚类处理的 Commit 数: {opt_config.SIMPLE_COMMITS_PER_CLUSTER}")
    print_info(f"聚类分析最大并行数: {opt_config.SIMPLE_CLUSTER_MAX_WORKERS}")
    print_info(f"Commit 分析最大并行数: {opt_config.SIMPLE_COMMIT_MAX_WORKERS}")
    print_info(f"每个 Commit 生成的规则数: {opt_config.COMMIT_GENERATION_COUNT}")
    print_info(f"详细输出模式: {opt_config.VERBOSE}")
    print_info(f"{'='*80}")
    print_info("")
    
    # 确认操作
    print_info("⚠️  警告: 此操作将删除所有错误的 Semgrep 规则文件（YAML 和 JSON）")
    print_info("   只删除已生成规则但JSON结果包含fatal/error错误的规则")
    print_info("   不包括未生成的规则（both_missing, yaml_missing, json_missing）")
    print_info("")
    
    result = cleanup_cluster_file(CLUSTER_JSON_PATH)
    
    if result:
        stats = result['overall_statistics']
        print_info(f"\n清理完成:")
        print_info(f"  共检查 {stats['total_commits_checked']} 个 commit")
        print_info(f"  发现 {stats['total_commits_with_errors']} 个 commit 有错误规则")
        print_info(f"  共删除 {stats['total_rules_deleted']} 个错误规则")
        print_info(f"\n现在可以重新运行 process_cluster.py 来生成缺失的规则")

if __name__ == "__main__":
    main()

