import os
import json
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from import_configs import global_config, generate_config

def print_info(message):
    """关键信息输出 - 始终显示"""
    print(message, flush=True)

def print_verbose(message):
    """详细信息输出 - 仅在VERBOSE模式下显示"""
    if generate_config.VERBOSE:
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
    knowledge_base_path = os.path.join(global_config.root_path, generate_config.KNOWLEDGE_BASE_RELATIVE_PATH)
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

def count_existing_semgrep_rules(semgrep_dir):
    """统计已存在的semgrep规则数量（所有规则文件，不限制编号）"""
    if not os.path.exists(semgrep_dir):
        return 0
    
    rule_files = glob.glob(os.path.join(semgrep_dir, "*.yml")) + \
                 glob.glob(os.path.join(semgrep_dir, "*.yaml"))
    return len(rule_files)

def count_valid_semgrep_rules(semgrep_dir):
    """统计编号1-5范围内的有效semgrep规则数量（与process_cluster.py保持一致）"""
    if not os.path.exists(semgrep_dir):
        return 0
    
    valid_rules_count = 0
    
    # 检查1到COMMIT_GENERATION_COUNT范围内的规则文件
    for rule_number in range(1, generate_config.COMMIT_GENERATION_COUNT + 1):
        yaml_path = os.path.join(semgrep_dir, f"{rule_number}.yaml")
        yml_path = os.path.join(semgrep_dir, f"{rule_number}.yml")
        
        # 检查文件是否存在且非空
        rule_file = None
        if os.path.exists(yaml_path) and os.path.getsize(yaml_path) > 0:
            rule_file = yaml_path
        elif os.path.exists(yml_path) and os.path.getsize(yml_path) > 0:
            rule_file = yml_path
        
        if rule_file:
            # 验证文件内容是否包含基本的semgrep规则关键字
            try:
                with open(rule_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content and ('rules:' in content or 'rule:' in content):
                        valid_rules_count += 1
            except:
                continue
    
    return valid_rules_count

def get_existing_rule_numbers(semgrep_dir):
    """获取实际存在的规则文件编号列表"""
    if not os.path.exists(semgrep_dir):
        return []
    
    rule_numbers = []
    # 检查1到COMMIT_GENERATION_COUNT范围内的规则文件
    for rule_number in range(1, generate_config.COMMIT_GENERATION_COUNT + 1):
        yaml_path = os.path.join(semgrep_dir, f"{rule_number}.yaml")
        yml_path = os.path.join(semgrep_dir, f"{rule_number}.yml")
        if (os.path.exists(yaml_path) and os.path.getsize(yaml_path) > 0) or \
           (os.path.exists(yml_path) and os.path.getsize(yml_path) > 0):
            rule_numbers.append(rule_number)
    
    return sorted(rule_numbers)

def has_valid_semgrep_rules(commit_info):
    """检查commit是否已经生成过足够数量的有效semgrep规则"""
    try:
        paths = build_commit_paths(commit_info)
        semgrep_dir = paths['semgrep_dir']
        
        # 检查semgrep目录是否存在
        if not os.path.exists(semgrep_dir):
            return False
        
        # 检查规则文件数量是否达到要求
        # 期望每个commit有COMMIT_GENERATION_COUNT个规则文件
        valid_rules_count = 0
        
        # 检查1到COMMIT_GENERATION_COUNT范围内的规则文件
        for rule_number in range(1, generate_config.COMMIT_GENERATION_COUNT + 1):
            yaml_path = os.path.join(semgrep_dir, f"{rule_number}.yaml")
            yml_path = os.path.join(semgrep_dir, f"{rule_number}.yml")
            
            # 检查文件是否存在且非空
            rule_file = None
            if os.path.exists(yaml_path) and os.path.getsize(yaml_path) > 0:
                rule_file = yaml_path
            elif os.path.exists(yml_path) and os.path.getsize(yml_path) > 0:
                rule_file = yml_path
            
            if rule_file:
                # 验证文件内容是否包含基本的semgrep规则关键字
                try:
                    with open(rule_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content and ('rules:' in content or 'rule:' in content):
                            valid_rules_count += 1
                except:
                    continue
        
        # 只有当规则数量达到COMMIT_GENERATION_COUNT时才认为有足够的规则
        return valid_rules_count >= generate_config.COMMIT_GENERATION_COUNT
    except Exception as e:
        print_verbose(f"Error checking semgrep rules for {commit_info.get('hash', 'unknown')}: {e}")
        return False

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

# ==================== 功能1: 分析已有规则状态 ====================

def analyze_single_cluster_status(cluster_info, cluster_index, total_clusters, commits_needed_per_cluster):
    """分析单个聚类的semgrep生成状态"""
    cluster_id = cluster_info.get('cluster_id', f"cluster_{cluster_index}")
    
    cluster_analysis = {
        "cluster_id": cluster_id,
        "cluster_index": cluster_index,
        "total_commits": len(cluster_info['commits']),
        "processable_commits": 0,
        "unprocessable_commits": 0,
        "commits_with_semgrep_rules": 0,
        "commits_without_semgrep_rules": 0,
        "commits_needed_per_cluster": commits_needed_per_cluster,
        "commits_still_needed": 0,
        "cluster_requirement_met": False,
        "total_existing_rules": 0,
        "commit_details": []
    }
    
    commits_with_rules = 0
    total_existing_rules = 0
    
    # 筛选可处理的commits
    processable_commits = []
    for commit in cluster_info['commits']:
        processable, error_msg = is_commit_processable(commit)
        if processable:
            processable_commits.append(commit)
        else:
            cluster_analysis["unprocessable_commits"] += 1
    
    # 选择前n个可处理的commits（如果总数量不够n的话就考虑所有commit）
    target_count = commits_needed_per_cluster
    top_n_commits = processable_commits[:target_count] if len(processable_commits) >= target_count else processable_commits
    
    # 只统计前n个可处理的commits
    for commit in top_n_commits:
        commit_hash = commit['hash']
        repo_name = commit['repository_name']
        
        commit_detail = {
            "repository_name": repo_name,
            "hash": commit_hash,
            "is_processable": True,
            "processable_error": None,
            "has_semgrep_rules": False,
            "existing_rules_count": 0
        }
        
        cluster_analysis["processable_commits"] += 1
        has_rules = has_valid_semgrep_rules(commit)
        commit_detail["has_semgrep_rules"] = has_rules
        
        if has_rules:
            commits_with_rules += 1
            commit_paths = build_commit_paths(commit)
            rules_count = count_valid_semgrep_rules(commit_paths['semgrep_dir'])
            commit_detail["existing_rules_count"] = rules_count
            total_existing_rules += rules_count
        
        cluster_analysis["commit_details"].append(commit_detail)
    
    cluster_analysis["commits_with_semgrep_rules"] = commits_with_rules
    cluster_analysis["commits_without_semgrep_rules"] = cluster_analysis["processable_commits"] - commits_with_rules
    cluster_analysis["total_existing_rules"] = total_existing_rules
    
    # 实际需要的commit数：如果processable_commits少于要求数，则使用processable_commits
    actual_commits_needed = min(commits_needed_per_cluster, cluster_analysis["processable_commits"])
    cluster_analysis["actual_commits_needed"] = actual_commits_needed
    cluster_analysis["commits_still_needed"] = max(0, actual_commits_needed - commits_with_rules)
    cluster_analysis["cluster_requirement_met"] = commits_with_rules >= actual_commits_needed
    
    return cluster_analysis

# ==================== 功能2: 分析处理需求 ====================

def analyze_single_cluster_needs(cluster_info, cluster_index, total_clusters, commits_needed_per_cluster):
    """分析单个聚类需要生成semgrep规则的commit数量"""
    cluster_id = cluster_info.get('cluster_id', f"cluster_{cluster_index}")
    
    cluster_analysis = {
        "cluster_id": cluster_id,
        "cluster_index": cluster_index,
        "total_commits": len(cluster_info['commits']),
        "processable_commits": 0,
        "unprocessable_commits": 0,
        "commits_needed_per_cluster": commits_needed_per_cluster,
        "commits_to_process": 0,
        "processable_commit_details": []
    }
    
    # 筛选可处理的commits
    all_processable_commits = []
    for commit in cluster_info['commits']:
        processable, error_msg = is_commit_processable(commit)
        if processable:
            all_processable_commits.append(commit)
        else:
            cluster_analysis["unprocessable_commits"] += 1
    
    # 选择前n个可处理的commits（如果总数量不够n的话就考虑所有commit）
    target_count = commits_needed_per_cluster
    top_n_commits = all_processable_commits[:target_count] if len(all_processable_commits) >= target_count else all_processable_commits
    
    processable_commits = []
    commits_without_rules = []
    
    # 只统计前n个可处理的commits
    for commit in top_n_commits:
        cluster_analysis["processable_commits"] += 1
        processable_commits.append({
            "repository_name": commit['repository_name'],
            "hash": commit['hash']
        })
        # 检查是否有足够的规则，如果没有则加入待处理列表
        if not has_valid_semgrep_rules(commit):
            commits_without_rules.append({
                "repository_name": commit['repository_name'],
                "hash": commit['hash']
            })
    
    # 实际需要的commit数：如果processable_commits少于要求数，则使用processable_commits
    actual_commits_needed = min(commits_needed_per_cluster, len(processable_commits))
    # 计算还需要多少个commit有规则
    commits_with_rules = len(processable_commits) - len(commits_without_rules)
    commits_still_needed = max(0, actual_commits_needed - commits_with_rules)
    
    # 需要处理的commit：优先处理没有规则的commit，如果还不够则从有规则的commit中选择
    if commits_still_needed > 0:
        commits_to_process = min(commits_still_needed, len(commits_without_rules))
        cluster_analysis["commits_to_process"] = commits_to_process
        cluster_analysis["processable_commit_details"] = commits_without_rules[:commits_to_process]
    else:
        cluster_analysis["commits_to_process"] = 0
        cluster_analysis["processable_commit_details"] = []
    
    return cluster_analysis

# ==================== 功能3: 分析规则质量 ====================

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

def analyze_single_commit_rules(commit_info, cluster_id, commit_idx, total_commits_in_cluster):
    """分析单个commit的实际已生成规则（只分析存在的规则文件）"""
    commit_hash = commit_info['hash']
    repo_name = commit_info['repository_name']
    
    commit_analysis = {
        "repository_name": repo_name,
        "hash": commit_hash,
        "cluster_id": cluster_id,
        "expected_rules": generate_config.COMMIT_GENERATION_COUNT,  # 期望的规则数
        "actual_rules_count": 0,  # 实际已生成的规则数
        "rule_analyses": [],
        "summary": {
            "usable_rules": 0,
            "needs_fixing_rules": 0,
            "yaml_missing": 0,
            "json_missing": 0,
            "both_missing": 0,
            "json_parse_errors": 0
        }
    }
    
    try:
        paths = build_commit_paths(commit_info)
        semgrep_dir = paths['semgrep_dir']
        
        # 只分析实际存在的规则文件
        existing_rule_numbers = get_existing_rule_numbers(semgrep_dir)
        commit_analysis["actual_rules_count"] = len(existing_rule_numbers)
        
        for rule_number in existing_rule_numbers:
            rule_analysis = analyze_single_rule(rule_number, semgrep_dir)
            commit_analysis["rule_analyses"].append(rule_analysis)
            
            status = rule_analysis["status"]
            if status == "usable":
                commit_analysis["summary"]["usable_rules"] += 1
            elif status == "needs_fixing":
                commit_analysis["summary"]["needs_fixing_rules"] += 1
            elif status == "yaml_missing":
                commit_analysis["summary"]["yaml_missing"] += 1
            elif status == "json_missing":
                commit_analysis["summary"]["json_missing"] += 1
            elif status == "both_missing":
                commit_analysis["summary"]["both_missing"] += 1
            elif status == "json_parse_error":
                commit_analysis["summary"]["json_parse_errors"] += 1
        
        return True, commit_analysis
    except Exception as e:
        commit_analysis["error_message"] = str(e)
        return False, commit_analysis

def analyze_single_cluster_rules(cluster_info, cluster_index, total_clusters, commits_needed_per_cluster):
    """分析单个聚类中需要生成规则的commit的规则质量"""
    cluster_id = cluster_info.get('cluster_id', f"cluster_{cluster_index}")
    
    cluster_analysis = {
        "cluster_id": cluster_id,
        "cluster_index": cluster_index,
        "total_commits": len(cluster_info['commits']),
        "processable_commits": 0,
        "commits_needed": 0,  # 需要生成规则的commit数
        "commits_with_rules": 0,  # 已有规则的commit数
        "processed_commits": 0,  # 已分析的commit数
        "expected_rules": 0,  # 期望的规则总数（需要生成规则的commit数 * 每个commit的规则数）
        "actual_rules": 0,  # 实际已生成的规则总数
        "commit_analyses": [],
        "summary": {
            "usable_rules": 0,
            "needs_fixing_rules": 0,
            "yaml_missing": 0,
            "json_missing": 0,
            "both_missing": 0,
            "json_parse_errors": 0
        }
    }
    
    # 筛选可处理的commits
    all_processable_commits = []
    for commit in cluster_info['commits']:
        processable, error_msg = is_commit_processable(commit)
        if processable:
            all_processable_commits.append(commit)
    
    # 选择前n个可处理的commits（如果总数量不够n的话就考虑所有commit）
    target_count = commits_needed_per_cluster
    top_n_commits = all_processable_commits[:target_count] if len(all_processable_commits) >= target_count else all_processable_commits
    
    processable_commits = []
    commits_with_rules = []
    
    # 只统计前n个可处理的commits
    for commit in top_n_commits:
        processable_commits.append(commit)
        if has_valid_semgrep_rules(commit):
            commits_with_rules.append(commit)
    
    cluster_analysis["processable_commits"] = len(processable_commits)
    cluster_analysis["commits_with_rules"] = len(commits_with_rules)
    
    # 计算需要生成规则的commit数
    actual_commits_needed = min(commits_needed_per_cluster, len(processable_commits))
    cluster_analysis["commits_needed"] = actual_commits_needed
    
    # 期望的规则总数 = 需要生成规则的commit数 * 每个commit的规则数
    cluster_analysis["expected_rules"] = actual_commits_needed * generate_config.COMMIT_GENERATION_COUNT
    
    if not commits_with_rules:
        return cluster_analysis
    
    # 只分析已经有规则的commit（这些是需要生成规则的commit中的一部分）
    commits_to_analyze = commits_with_rules[:actual_commits_needed]  # 只分析前N个（符合要求）
    total_commits_in_cluster = len(commits_to_analyze)
    
    # 计算实际已生成的规则总数（只统计编号1-5范围内的有效规则）
    total_actual_rules = 0
    for commit in commits_to_analyze:
        commit_paths = build_commit_paths(commit)
        total_actual_rules += count_valid_semgrep_rules(commit_paths['semgrep_dir'])
    
    cluster_analysis["actual_rules"] = total_actual_rules
    
    with ThreadPoolExecutor(max_workers=generate_config.SIMPLE_COMMIT_MAX_WORKERS) as executor:
        future_to_commit = {}
        for commit_idx, commit in enumerate(commits_to_analyze, 1):
            future = executor.submit(analyze_single_commit_rules, commit, cluster_id, commit_idx, total_commits_in_cluster)
            future_to_commit[future] = commit
        
        for future in as_completed(future_to_commit):
            success, commit_analysis = future.result()
            
            if success:
                cluster_analysis["commit_analyses"].append(commit_analysis)
                cluster_analysis["processed_commits"] += 1
                
                for key in cluster_analysis["summary"]:
                    cluster_analysis["summary"][key] += commit_analysis["summary"][key]
    
    return cluster_analysis

def analyze_single_cluster_all(cluster_info, cluster_index, total_clusters, commits_needed_per_cluster):
    """执行单个聚类的所有三种分析"""
    status_result = analyze_single_cluster_status(cluster_info, cluster_index, total_clusters, commits_needed_per_cluster)
    needs_result = analyze_single_cluster_needs(cluster_info, cluster_index, total_clusters, commits_needed_per_cluster)
    quality_result = analyze_single_cluster_rules(cluster_info, cluster_index, total_clusters, commits_needed_per_cluster)
    
    return {
        "cluster_id": status_result["cluster_id"],
        "cluster_index": cluster_index,
        "status_analysis": status_result,
        "needs_analysis": needs_result,
        "quality_analysis": quality_result
    }

# ==================== 并行分析函数 ====================

def analyze_clusters_parallel(clusters, commits_needed_per_cluster, max_workers=None):
    """并行分析多个聚类的所有统计信息"""
    if max_workers is None:
        max_workers = getattr(generate_config, 'SIMPLE_CLUSTER_MAX_WORKERS', 4)
    
    results = []
    clusters_to_analyze = clusters
    
    if hasattr(generate_config, 'SIMPLE_PROCESS_CLUSTER_LIMIT') and generate_config.SIMPLE_PROCESS_CLUSTER_LIMIT > 0:
        clusters_to_analyze = clusters[:generate_config.SIMPLE_PROCESS_CLUSTER_LIMIT]
    
    if not clusters_to_analyze:
        return results
    
    total_clusters = len(clusters_to_analyze)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cluster = {}
        for i, cluster in enumerate(clusters_to_analyze):
            future = executor.submit(analyze_single_cluster_all, cluster, i, total_clusters, commits_needed_per_cluster)
            future_to_cluster[future] = i
        
        for future in as_completed(future_to_cluster):
            cluster_idx = future_to_cluster[future]
            try:
                cluster_result = future.result()
                results.append(cluster_result)
            except Exception as e:
                error_result = {
                    "cluster_id": f"cluster_{cluster_idx}",
                    "cluster_index": cluster_idx,
                    "error_message": f"Cluster analysis failed: {str(e)}"
                }
                results.append(error_result)
    
    results.sort(key=lambda x: x['cluster_index'])
    return results

# ==================== 统计生成函数 ====================

def generate_status_statistics(valid_clusters, total_clusters):
    """生成状态统计信息"""
    total_commits = sum(r['total_commits'] for r in valid_clusters)
    processable_commits = sum(r['processable_commits'] for r in valid_clusters)
    unprocessable_commits = sum(r['unprocessable_commits'] for r in valid_clusters)
    
    clusters_meeting_requirement = sum(1 for r in valid_clusters if r.get('cluster_requirement_met', False))
    clusters_not_meeting_requirement = len(valid_clusters) - clusters_meeting_requirement
    
    commits_with_semgrep_rules = sum(r['commits_with_semgrep_rules'] for r in valid_clusters)
    commits_without_semgrep_rules = sum(r['commits_without_semgrep_rules'] for r in valid_clusters)
    total_existing_rules = sum(r['total_existing_rules'] for r in valid_clusters)
    total_commits_still_needed = sum(r['commits_still_needed'] for r in valid_clusters)
    
    commits_needed_per_cluster = valid_clusters[0]['commits_needed_per_cluster'] if valid_clusters else 3
    
    return {
        "total_clusters": total_clusters,
        "valid_clusters": len(valid_clusters),
        "failed_clusters": total_clusters - len(valid_clusters),
        "commits_needed_per_cluster": commits_needed_per_cluster,
        "clusters_meeting_requirement": clusters_meeting_requirement,
        "clusters_not_meeting_requirement": clusters_not_meeting_requirement,
        "cluster_completion_rate": clusters_meeting_requirement / len(valid_clusters) if valid_clusters else 0,
        "total_commits": total_commits,
        "processable_commits": processable_commits,
        "unprocessable_commits": unprocessable_commits,
        "commits_with_semgrep_rules": commits_with_semgrep_rules,
        "commits_without_semgrep_rules": commits_without_semgrep_rules,
        "total_existing_rules": total_existing_rules,
        "total_commits_still_needed": total_commits_still_needed,
        "estimated_rules_to_generate": total_commits_still_needed * getattr(generate_config, 'COMMIT_GENERATION_COUNT', 5),
        "processable_rate": processable_commits / total_commits if total_commits > 0 else 0,
        "commits_with_rules_rate": commits_with_semgrep_rules / processable_commits if processable_commits > 0 else 0
    }

def generate_needs_statistics(valid_clusters, total_clusters):
    """生成需求统计信息"""
    total_commits = sum(r['total_commits'] for r in valid_clusters)
    processable_commits = sum(r['processable_commits'] for r in valid_clusters)
    unprocessable_commits = sum(r['unprocessable_commits'] for r in valid_clusters)
    
    clusters_need_processing = sum(1 for r in valid_clusters if r.get('commits_to_process', 0) > 0)
    total_commits_to_process = sum(r.get('commits_to_process', 0) for r in valid_clusters)
    
    commits_needed_per_cluster = valid_clusters[0].get('commits_needed_per_cluster', 3) if valid_clusters else 3
    estimated_rules_to_generate = total_commits_to_process * getattr(generate_config, 'COMMIT_GENERATION_COUNT', 5)
    
    return {
        "total_clusters": total_clusters,
        "valid_clusters": len(valid_clusters),
        "failed_clusters": total_clusters - len(valid_clusters),
        "commits_needed_per_cluster": commits_needed_per_cluster,
        "clusters_need_processing": clusters_need_processing,
        "total_commits_to_process": total_commits_to_process,
        "estimated_rules_to_generate": estimated_rules_to_generate,
        "total_commits": total_commits,
        "processable_commits": processable_commits,
        "unprocessable_commits": unprocessable_commits,
        "processable_rate": processable_commits / total_commits if total_commits > 0 else 0,
    }

def generate_quality_statistics(valid_clusters, total_clusters):
    """生成规则质量统计信息"""
    total_commits = sum(r.get('total_commits', 0) for r in valid_clusters)
    processable_commits = sum(r.get('processable_commits', 0) for r in valid_clusters)
    processed_commits = sum(r.get('processed_commits', 0) for r in valid_clusters)
    
    # 期望的规则总数 = 所有聚类需要生成规则的commit数 * 每个commit的规则数
    total_commits_needed = sum(r.get('commits_needed', 0) for r in valid_clusters)
    expected_rules = total_commits_needed * getattr(generate_config, 'COMMIT_GENERATION_COUNT', 5)
    
    # 实际已生成的规则总数
    actual_rules = sum(r.get('actual_rules', 0) for r in valid_clusters)
    
    overall_summary = {
        "usable_rules": 0,
        "needs_fixing_rules": 0,
        "yaml_missing": 0,
        "json_missing": 0,
        "both_missing": 0,
        "json_parse_errors": 0
    }
    
    for cluster in valid_clusters:
        summary = cluster.get("summary", {})
        for key in overall_summary:
            overall_summary[key] += summary.get(key, 0)
    
    total_problematic_rules = (overall_summary["needs_fixing_rules"] + 
                              overall_summary["yaml_missing"] + 
                              overall_summary["json_missing"] + 
                              overall_summary["both_missing"] + 
                              overall_summary["json_parse_errors"])
    
    total_file_missing_rules = (overall_summary["yaml_missing"] + 
                               overall_summary["json_missing"] + 
                               overall_summary["both_missing"])
    
    # 缺失的规则数 = 期望的规则数 - 实际已生成的规则数
    missing_rules = max(0, expected_rules - actual_rules)
    
    return {
        "total_clusters": total_clusters,
        "valid_clusters": len(valid_clusters),
        "failed_clusters": total_clusters - len(valid_clusters),
        "total_commits": total_commits,
        "processable_commits": processable_commits,
        "processed_commits": processed_commits,
        "processable_rate": processable_commits / total_commits if total_commits > 0 else 0,
        "total_commits_needed": total_commits_needed,  # 需要生成规则的commit总数
        "expected_rules": expected_rules,  # 期望的规则总数
        "actual_rules": actual_rules,  # 实际已生成的规则总数
        "missing_rules": missing_rules,  # 缺失的规则数（还未生成）
        "usable_rules": overall_summary["usable_rules"],  # 已生成规则中可用的数量
        "needs_fixing_rules": overall_summary["needs_fixing_rules"],  # 已生成规则中需要修复的数量
        "total_problematic_rules": total_problematic_rules,  # 已生成规则中有问题的数量
        "total_file_missing_rules": total_file_missing_rules,  # 已生成规则中文件缺失的数量
        "yaml_missing": overall_summary["yaml_missing"],
        "json_missing": overall_summary["json_missing"],
        "both_missing": overall_summary["both_missing"],
        "json_parse_errors": overall_summary["json_parse_errors"],
        "usable_rate": overall_summary["usable_rules"] / actual_rules if actual_rules > 0 else 0,  # 已生成规则中可用的比例
        "needs_fixing_rate": overall_summary["needs_fixing_rules"] / actual_rules if actual_rules > 0 else 0,  # 已生成规则中需要修复的比例
        "file_missing_rate": total_file_missing_rules / actual_rules if actual_rules > 0 else 0,  # 已生成规则中文件缺失的比例
        "overall_success_rate": overall_summary["usable_rules"] / expected_rules if expected_rules > 0 else 0,  # 整体成功率（可用规则/期望规则）
        "generation_progress": actual_rules / expected_rules if expected_rules > 0 else 0  # 生成进度（实际规则/期望规则）
    }

def generate_overall_statistics(cluster_results):
    """生成整体统计信息（合并重复字段）"""
    total_clusters = len(cluster_results)
    valid_clusters = [r for r in cluster_results if 'error_message' not in r]
    
    status_stats = generate_status_statistics(
        [r.get("status_analysis", r) for r in valid_clusters], total_clusters
    )
    needs_stats = generate_needs_statistics(
        [r.get("needs_analysis", r) for r in valid_clusters], total_clusters
    )
    quality_stats = generate_quality_statistics(
        [r.get("quality_analysis", r) for r in valid_clusters], total_clusters
    )
    
    # 合并重复字段，使用status_statistics中的值（因为更完整）
    merged_stats = {
        # 基础信息（合并重复）
        "total_clusters": status_stats["total_clusters"],
        "valid_clusters": status_stats["valid_clusters"],
        "failed_clusters": status_stats["failed_clusters"],
        "total_commits": status_stats["total_commits"],
        "processable_commits": status_stats["processable_commits"],
        "unprocessable_commits": status_stats["unprocessable_commits"],
        "processable_rate": status_stats["processable_rate"],
        
        # 配置信息
        "commits_needed_per_cluster": status_stats["commits_needed_per_cluster"],
        
        # 状态统计
        "clusters_meeting_requirement": status_stats["clusters_meeting_requirement"],
        "clusters_not_meeting_requirement": status_stats["clusters_not_meeting_requirement"],
        "cluster_completion_rate": status_stats["cluster_completion_rate"],
        "commits_with_semgrep_rules": status_stats["commits_with_semgrep_rules"],
        "commits_without_semgrep_rules": status_stats["commits_without_semgrep_rules"],
        "commits_with_rules_rate": status_stats["commits_with_rules_rate"],
        "total_existing_rules": status_stats["total_existing_rules"],
        "total_commits_still_needed": status_stats["total_commits_still_needed"],
        "estimated_rules_to_generate_status": status_stats["estimated_rules_to_generate"],
        
        # 需求统计
        "clusters_need_processing": needs_stats["clusters_need_processing"],
        "total_commits_to_process": needs_stats["total_commits_to_process"],
        "estimated_rules_to_generate_needs": needs_stats["estimated_rules_to_generate"],
        
        # 质量统计
        "processed_commits": quality_stats["processed_commits"],
        "total_commits_needed": quality_stats["total_commits_needed"],
        "expected_rules": quality_stats["expected_rules"],
        "actual_rules": quality_stats["actual_rules"],
        "missing_rules": quality_stats["missing_rules"],
        "usable_rules": quality_stats["usable_rules"],
        "needs_fixing_rules": quality_stats["needs_fixing_rules"],
        "usable_rate": quality_stats["usable_rate"],
        "needs_fixing_rate": quality_stats["needs_fixing_rate"],
        "total_file_missing_rules": quality_stats["total_file_missing_rules"],
        "file_missing_rate": quality_stats["file_missing_rate"],
        "yaml_missing": quality_stats["yaml_missing"],
        "json_missing": quality_stats["json_missing"],
        "both_missing": quality_stats["both_missing"],
        "json_parse_errors": quality_stats["json_parse_errors"],
        "total_problematic_rules": quality_stats["total_problematic_rules"],
        "overall_success_rate": quality_stats["overall_success_rate"],
        "generation_progress": quality_stats["generation_progress"],
    }
    
    return merged_stats

# ==================== 输出函数 ====================

def print_comprehensive_statistics(overall_statistics, commits_needed_per_cluster):
    """打印综合统计信息（合并重复，使用中文）"""
    print_info(f"\n{'='*80}")
    print_info(f"SEMGREP 规则生成综合分析报告")
    print_info(f"{'='*80}")
    print_info(f"目标配置: 每个聚类需要 {commits_needed_per_cluster} 个有 semgrep 规则的 commit")
    print_info(f"")
    
    # 基础数据概览
    print_info(f"【基础数据概览】")
    print_info(f"  聚类总数: {overall_statistics['total_clusters']} (有效: {overall_statistics['valid_clusters']}, 失败: {overall_statistics['failed_clusters']})")
    print_info(f"  Commit 总数: {overall_statistics['total_commits']}")
    print_info(f"  可处理 Commit: {overall_statistics['processable_commits']} ({overall_statistics['processable_rate']:.1%})")
    print_info(f"  不可处理 Commit: {overall_statistics['unprocessable_commits']}")
    print_info(f"")
    
    # 聚类完成状态
    print_info(f"【聚类完成状态】")
    print_info(f"  满足要求的聚类: {overall_statistics['clusters_meeting_requirement']} ({overall_statistics['cluster_completion_rate']:.1%})")
    print_info(f"  未满足要求的聚类: {overall_statistics['clusters_not_meeting_requirement']}")
    print_info(f"")
    
    # Commit 规则生成状态（合并重复数据）
    print_info(f"【Commit 规则生成状态】")
    print_info(f"  已有规则的 Commit: {overall_statistics['commits_with_semgrep_rules']} ({overall_statistics['commits_with_rules_rate']:.1%} 的可处理 commit)")
    print_info(f"  无规则的 Commit: {overall_statistics['commits_without_semgrep_rules']}")
    print_info(f"")
    
    # 规则生成进度（合并重复数据）
    print_info(f"【规则生成进度】")
    print_info(f"  需要生成规则的 Commit 总数: {overall_statistics.get('total_commits_needed', 0)}")
    print_info(f"  待处理的 Commit 数: {overall_statistics.get('total_commits_to_process', overall_statistics.get('total_commits_still_needed', 0))}")
    print_info(f"  期望的规则总数: {overall_statistics['expected_rules']} (需要生成规则的commit数 × {getattr(generate_config, 'COMMIT_GENERATION_COUNT', 5)})")
    print_info(f"  实际已生成的规则总数: {overall_statistics.get('actual_rules', overall_statistics.get('total_existing_rules', 0))}")
    print_info(f"  缺失的规则数（还未生成）: {overall_statistics.get('missing_rules', overall_statistics.get('estimated_rules_to_generate_needs', 0))}")
    print_info(f"  生成进度: {overall_statistics.get('generation_progress', 0):.1%}")
    if overall_statistics.get('clusters_need_processing', 0) > 0:
        print_info(f"  需要处理的聚类数: {overall_statistics['clusters_need_processing']}")
    print_info(f"")
    
    # 规则质量分析（只针对已生成的规则）
    actual_rules = overall_statistics.get('actual_rules', overall_statistics.get('total_existing_rules', 0))
    if actual_rules > 0:
        print_info(f"【已生成规则的质量分析】")
        print_info(f"  已分析的 Commit 数: {overall_statistics['processed_commits']}")
        print_info(f"")
        print_info(f"  规则质量统计（基于 {actual_rules} 个已生成的规则）:")
        print_info(f"    ✅ 可用规则（运行无报错）: {overall_statistics['usable_rules']} ({overall_statistics['usable_rate']:.1%})")
        print_info(f"    ❌ 需要修复（JSON 中有 fatal/error 级别错误）: {overall_statistics['needs_fixing_rules']} ({overall_statistics['needs_fixing_rate']:.1%})")
        print_info(f"    📂 文件缺失问题: {overall_statistics['total_file_missing_rules']} ({overall_statistics['file_missing_rate']:.1%})")
        
        # 只在有文件缺失问题时显示明细
        if overall_statistics['total_file_missing_rules'] > 0:
            print_info(f"")
            print_info(f"  文件缺失明细:")
            print_info(f"    YAML 文件缺失: {overall_statistics['yaml_missing']}")
            print_info(f"    JSON 文件缺失: {overall_statistics['json_missing']}")
            print_info(f"    两者都缺失: {overall_statistics['both_missing']}")
            if overall_statistics['json_parse_errors'] > 0:
                print_info(f"    JSON 解析错误: {overall_statistics['json_parse_errors']}")
        
        print_info(f"")
        print_info(f"  整体成功率（可用规则/期望规则）: {overall_statistics['overall_success_rate']:.1%}")
        if overall_statistics.get('total_problematic_rules', 0) > 0:
            print_info(f"  已生成规则中的问题规则数: {overall_statistics['total_problematic_rules']} ({(overall_statistics['total_problematic_rules']/actual_rules*100):.1f}%)")
    
    print_info(f"{'='*80}")

# ==================== 主分析函数 ====================

def save_analysis_result(result_data, output_path):
    """保存分析结果到JSON文件"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=getattr(generate_config, 'DEFAULT_OUTPUT_INDENT', 2), ensure_ascii=False)
        print_info(f"分析结果已保存到: {output_path}")
        return True
    except Exception as e:
        print_info(f"保存分析结果时出错: {e}")
        return False

def analyze_cluster_file(cluster_json_path):
    """综合分析聚类JSON文件（执行所有三种统计）"""
    start_time = time.time()
    
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
    
    commits_needed_per_cluster = generate_config.SIMPLE_COMMITS_PER_CLUSTER
    
    cluster_results = analyze_clusters_parallel(clusters, commits_needed_per_cluster)
    
    overall_statistics = generate_overall_statistics(cluster_results)
    
    # 输出合并后的综合统计信息（中文）
    print_comprehensive_statistics(overall_statistics, commits_needed_per_cluster)
    
    return {
        "overall_statistics": overall_statistics,
        "cluster_results": cluster_results
    }

# ==================== 主函数 ====================

def main():
    """主函数"""
    CLUSTER_JSON_PATH = global_config.root_path + "python/3-cluster_huawei_stage2/result_final/0_8_2_merged_order.json"
    
    result = analyze_cluster_file(CLUSTER_JSON_PATH)
    
    # 统计信息已在 print_comprehensive_statistics 中完整输出，此处不再重复

if __name__ == "__main__":
    main()
