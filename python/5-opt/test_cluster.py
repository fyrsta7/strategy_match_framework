import os
import json
import sys
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from get_semgrep_result_commit import process_all_semgrep_rules
from analyze_semgrep_result import analyze_commit_semgrep_coverage

# 全局锁，确保并行安全
_file_operation_lock = threading.Lock()

def find_before_file(commit_dir):
    """查找before文件（任意后缀）"""
    pattern = os.path.join(commit_dir, "before.*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None

def get_commit_knowledge_paths(commit_info, knowledge_base_path):
    """获取knowledge_base中的commit路径"""
    repo_name = commit_info['repository_name']
    commit_hash = commit_info['hash']
    
    commit_dir = os.path.join(knowledge_base_path, repo_name, 'modified_file', commit_hash)
    before_file = find_before_file(commit_dir)
    semgrep_dir = os.path.join(commit_dir, 'semgrep')
    
    return {
        'commit_dir': commit_dir,
        'before_file': before_file,
        'semgrep_dir': semgrep_dir
    }

def get_test_result_paths(commit1_hash, commit2_hash, cluster_id, root_path):
    """获取测试结果存储路径"""
    base_dir = os.path.join(root_path, 'semgrep_result', 'intermediate', f'cluster_{cluster_id}', commit1_hash)
    
    return {
        'base_dir': base_dir,
        'test_dir': os.path.join(base_dir, 'test_results', commit2_hash),
        'semgrep_raw_dir': os.path.join(base_dir, 'test_results', commit2_hash, 'semgrep_raw'),
        'analysis_file': os.path.join(base_dir, 'test_results', commit2_hash, 'analysis', 'analyze_result.json')
    }

def get_semgrep_rules(semgrep_dir):
    """获取commit的所有semgrep规则文件"""
    if not os.path.exists(semgrep_dir):
        return []
    
    yaml_files = glob.glob(os.path.join(semgrep_dir, "*.yaml"))
    yaml_files.extend(glob.glob(os.path.join(semgrep_dir, "*.yml")))
    return sorted(yaml_files)

def test_single_rule_on_commit(commit1_info, commit2_info, cluster_id, config):
    """测试commit1的所有规则在commit2上的表现"""
    commit1_hash = commit1_info['hash']
    commit2_hash = commit2_info['hash']
    
    # 获取路径
    commit1_paths = get_commit_knowledge_paths(commit1_info, config['knowledge_base_path'])
    commit2_paths = get_commit_knowledge_paths(commit2_info, config['knowledge_base_path'])
    test_paths = get_test_result_paths(commit1_hash, commit2_hash, cluster_id, config['root_path'])
    
    # 检查必要文件是否存在
    if not commit2_paths['before_file'] or not os.path.exists(commit2_paths['before_file']):
        return False, "commit2 before file not found"
    
    if not os.path.exists(commit1_paths['semgrep_dir']):
        return False, "commit1 semgrep rules not found"
    
    # 检查是否复用现有结果
    if config['reuse_existing'] and os.path.exists(test_paths['analysis_file']):
        try:
            with open(test_paths['analysis_file'], 'r', encoding='utf-8') as f:
                json.load(f)
            return True, "reused existing result"
        except:
            pass
    
    # 创建输出目录
    os.makedirs(test_paths['semgrep_raw_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(test_paths['analysis_file']), exist_ok=True)
    
    try:
        # 1. 运行semgrep规则
        total_count, success_count, _, _, _ = process_all_semgrep_rules(
            yaml_dir_path=commit1_paths['semgrep_dir'],
            target_file_path=commit2_paths['before_file'],
            json_output_dir=test_paths['semgrep_raw_dir'],
            skip_existing=config['reuse_existing']
        )
        
        if success_count == 0:
            return False, f"no rules executed successfully ({total_count} total)"
        
        # 2. 分析结果
        target_start_line = commit2_info['file_start_line']
        target_end_line = commit2_info['file_end_line']
        
        has_matching_rules = analyze_commit_semgrep_coverage(
            semgrep_results_dir=test_paths['semgrep_raw_dir'],
            target_start_line=target_start_line,
            target_end_line=target_end_line,
            coverage_config=config['coverage_config']
        )
        
        return True, f"completed: {success_count}/{total_count} rules, match={has_matching_rules}"
        
    except Exception as e:
        return False, f"execution error: {str(e)}"

def test_commit_rules_on_cluster(commit1_info, cluster_commits, cluster_id, config):
    """测试commit1的规则在整个聚类上的表现"""
    commit1_hash = commit1_info['hash']
    
    # 筛选可测试的commits
    testable_commits = []
    for commit in cluster_commits:
        if all(field in commit for field in ['file_start_line', 'file_end_line']):
            testable_commits.append(commit)
    
    if not testable_commits:
        return {
            'commit_hash': commit1_hash,
            'cluster_id': cluster_id,
            'total_tests': 0,
            'successful_tests': 0,
            'test_details': [],
            'error': 'no testable commits in cluster'
        }
    
    # 串行测试每个commit
    test_results = []
    successful_tests = 0
    
    for commit2 in testable_commits:
        commit2_hash = commit2['hash']
        
        success, message = test_single_rule_on_commit(commit1_info, commit2, cluster_id, config)
        
        test_detail = {
            'commit2_hash': commit2_hash,
            'success': success,
            'message': message
        }
        test_results.append(test_detail)
        
        if success:
            successful_tests += 1
    
    return {
        'commit_hash': commit1_hash,
        'cluster_id': cluster_id,
        'total_tests': len(testable_commits),
        'successful_tests': successful_tests,
        'test_details': test_results,
        'error': None
    }

def aggregate_rule_performance(commit1_hash, cluster_id, config):
    """汇总单个commit所有规则的性能"""
    test_base_dir = os.path.join(config['root_path'], 'semgrep_result', 'intermediate', f'cluster_{cluster_id}', commit1_hash, 'test_results')
    
    if not os.path.exists(test_base_dir):
        return {
            'commit_hash': commit1_hash,
            'cluster_id': cluster_id,
            'rules_performance': {},
            'summary': {
                'total_rules': 0,
                'rules_with_good_performance': 0,
                'best_rule_success_rate': 0.0
            }
        }
    
    # 获取所有测试结果
    commit_dirs = [d for d in os.listdir(test_base_dir) if os.path.isdir(os.path.join(test_base_dir, d))]
    
    rules_performance = {}
    total_tests = len(commit_dirs)
    
    # 收集每条规则的表现
    for commit2_hash in commit_dirs:
        analysis_file = os.path.join(test_base_dir, commit2_hash, 'analysis', 'analyze_result.json')
        
        if not os.path.exists(analysis_file):
            continue
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            matching_rules = analysis_data.get('matching_rules', [])
            
            for rule_info in matching_rules:
                rule_file = rule_info.get('rule_file', 'unknown')
                
                if rule_file not in rules_performance:
                    rules_performance[rule_file] = {
                        'total_tests': 0,
                        'successful_tests': 0,
                        'success_rate': 0.0,
                        'test_details': []
                    }
                
                rules_performance[rule_file]['total_tests'] += 1
                rules_performance[rule_file]['successful_tests'] += 1
                rules_performance[rule_file]['test_details'].append({
                    'commit2_hash': commit2_hash,
                    'passed': True
                })
            
            # 为没有匹配的规则也添加测试记录
            semgrep_raw_dir = os.path.join(test_base_dir, commit2_hash, 'semgrep_raw')
            if os.path.exists(semgrep_raw_dir):
                all_rule_files = [f for f in os.listdir(semgrep_raw_dir) if f.endswith('.json') and f != 'analyze_result.json']
                
                for rule_file in all_rule_files:
                    if rule_file not in [r.get('rule_file') for r in matching_rules]:
                        if rule_file not in rules_performance:
                            rules_performance[rule_file] = {
                                'total_tests': 0,
                                'successful_tests': 0,
                                'success_rate': 0.0,
                                'test_details': []
                            }
                        
                        rules_performance[rule_file]['total_tests'] += 1
                        rules_performance[rule_file]['test_details'].append({
                            'commit2_hash': commit2_hash,
                            'passed': False
                        })
        
        except Exception:
            continue
    
    # 计算成功率
    for rule_file in rules_performance:
        rule_data = rules_performance[rule_file]
        if rule_data['total_tests'] > 0:
            rule_data['success_rate'] = rule_data['successful_tests'] / rule_data['total_tests']
    
    # 生成汇总信息
    total_rules = len(rules_performance)
    rules_with_good_performance = sum(1 for r in rules_performance.values() if r['success_rate'] >= config['quality_threshold'])
    best_rule_success_rate = max([r['success_rate'] for r in rules_performance.values()], default=0.0)
    
    return {
        'commit_hash': commit1_hash,
        'cluster_id': cluster_id,
        'rules_performance': rules_performance,
        'summary': {
            'total_rules': total_rules,
            'rules_with_good_performance': rules_with_good_performance,
            'best_rule_success_rate': best_rule_success_rate
        }
    }

def process_single_cluster(cluster_info, cluster_idx, config):
    """处理单个聚类"""
    cluster_id = cluster_info.get('cluster_id', cluster_idx)
    commits = cluster_info.get('commits', [])
    
    # 找到有semgrep规则的commits
    commits_with_rules = []
    for commit in commits:
        if commit.get('semgrep_generation_success', False):
            commits_with_rules.append(commit)
    
    if not commits_with_rules:
        return {
            'cluster_id': cluster_id,
            'commits_tested': 0,
            'commits_with_good_rules': 0,
            'commit_results': [],
            'error': 'no commits with semgrep rules'
        }
    
    # 测试每个有规则的commit
    cluster_results = []
    commits_with_good_rules = 0
    
    for commit1 in commits_with_rules:
        # 1. 测试该commit的规则在聚类上的表现
        test_result = test_commit_rules_on_cluster(commit1, commits, cluster_id, config)
        
        # 2. 汇总该commit规则的性能
        performance_result = aggregate_rule_performance(commit1['hash'], cluster_id, config)
        
        # 3. 判断是否有好的规则
        has_good_rules = performance_result['summary']['rules_with_good_performance'] > 0
        if has_good_rules:
            commits_with_good_rules += 1
        
        commit_result = {
            'commit_hash': commit1['hash'],
            'test_result': test_result,
            'performance_result': performance_result,
            'has_good_rules': has_good_rules
        }
        cluster_results.append(commit_result)
    
    return {
        'cluster_id': cluster_id,
        'commits_tested': len(commits_with_rules),
        'commits_with_good_rules': commits_with_good_rules,
        'commit_results': cluster_results,
        'error': None
    }

def process_clusters_parallel(clusters, config):
    """并行处理多个聚类"""
    clusters_to_process = clusters
    if config['cluster_limit'] > 0:
        clusters_to_process = clusters[:config['cluster_limit']]
    
    if not clusters_to_process:
        return {}
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        future_to_cluster = {}
        
        for i, cluster in enumerate(clusters_to_process):
            future = executor.submit(process_single_cluster, cluster, i, config)
            future_to_cluster[future] = i
        
        with tqdm(total=len(future_to_cluster), desc="Processing clusters", unit="cluster") as pbar:
            for future in as_completed(future_to_cluster):
                cluster_idx = future_to_cluster[future]
                try:
                    result = future.result()
                    results[cluster_idx] = result
                    
                    # 更新进度条描述
                    cluster_id = result['cluster_id']
                    commits_tested = result['commits_tested']
                    commits_good = result['commits_with_good_rules']
                    pbar.set_postfix({
                        'cluster': cluster_id,
                        'tested': commits_tested,
                        'good': commits_good
                    })
                    
                except Exception as e:
                    results[cluster_idx] = {
                        'cluster_id': cluster_idx,
                        'commits_tested': 0,
                        'commits_with_good_rules': 0,
                        'commit_results': [],
                        'error': str(e)
                    }
                finally:
                    pbar.update(1)
    
    return results

def generate_summary_reports(cluster_results, config):
    """生成汇总报告"""
    summary_dir = os.path.join(config['root_path'], 'semgrep_result', 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # 全局统计
    total_clusters = len(cluster_results)
    clusters_with_good_rules = sum(1 for r in cluster_results.values() if r['commits_with_good_rules'] > 0)
    total_commits_tested = sum(r['commits_tested'] for r in cluster_results.values())
    total_commits_with_good_rules = sum(r['commits_with_good_rules'] for r in cluster_results.values())
    
    # 按commit汇总
    per_commit_summary = []
    for cluster_result in cluster_results.values():
        for commit_result in cluster_result['commit_results']:
            per_commit_summary.append({
                'cluster_id': cluster_result['cluster_id'],
                'commit_hash': commit_result['commit_hash'],
                'has_good_rules': commit_result['has_good_rules'],
                'rules_performance': commit_result['performance_result']['rules_performance'],
                'summary': commit_result['performance_result']['summary']
            })
    
    # 按聚类汇总
    per_cluster_summary = []
    for cluster_result in cluster_results.values():
        good_commits = [cr['commit_hash'] for cr in cluster_result['commit_results'] if cr['has_good_rules']]
        per_cluster_summary.append({
            'cluster_id': cluster_result['cluster_id'],
            'commits_tested': cluster_result['commits_tested'],
            'commits_with_good_rules': cluster_result['commits_with_good_rules'],
            'good_commits': good_commits
        })
    
    # 全局汇总
    global_summary = {
        'statistics': {
            'total_clusters': total_clusters,
            'clusters_with_good_rules': clusters_with_good_rules,
            'total_commits_tested': total_commits_tested,
            'commits_with_good_rules': total_commits_with_good_rules,
            'quality_threshold': config['quality_threshold']
        },
        'cluster_summary': per_cluster_summary
    }
    
    # 保存文件
    with _file_operation_lock:
        # 按commit汇总
        with open(os.path.join(summary_dir, 'per_commit_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(per_commit_summary, f, indent=2, ensure_ascii=False)
        
        # 按聚类汇总
        with open(os.path.join(summary_dir, 'per_cluster_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(per_cluster_summary, f, indent=2, ensure_ascii=False)
        
        # 全局汇总
        with open(os.path.join(summary_dir, 'global_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(global_summary, f, indent=2, ensure_ascii=False)
    
    return global_summary

def print_statistics(global_summary):
    """打印统计信息"""
    stats = global_summary['statistics']
    
    print(f"\n{'='*60}")
    print(f"TEST CLUSTER RESULTS - STATISTICS")
    print(f"{'='*60}")
    print(f"Quality threshold: {stats['quality_threshold']*100}%")
    print(f"Total clusters processed: {stats['total_clusters']}")
    print(f"Clusters with good rules: {stats['clusters_with_good_rules']} ({stats['clusters_with_good_rules']/stats['total_clusters']*100:.1f}%)")
    print(f"Total commits tested: {stats['total_commits_tested']}")
    print(f"Commits with good rules: {stats['commits_with_good_rules']} ({stats['commits_with_good_rules']/stats['total_commits_tested']*100:.1f}%)")
    print(f"{'='*60}")
    sys.stdout.flush()

def test_cluster_file(input_json_path, knowledge_base_path, root_path, 
                     cluster_limit=0, reuse_existing=True, quality_threshold=0.4, max_workers=16):
    """测试聚类文件的主函数"""
    
    # 验证输入
    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON file not found: {input_json_path}")
        return None
    
    # 读取聚类数据
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
    except Exception as e:
        print(f"Error reading input JSON: {e}")
        return None
    
    if 'clusters' not in cluster_data:
        print(f"Error: No 'clusters' field found in JSON")
        return None
    
    clusters = cluster_data['clusters']
    
    # 配置参数
    config = {
        'knowledge_base_path': knowledge_base_path,
        'root_path': root_path,
        'cluster_limit': cluster_limit,
        'reuse_existing': reuse_existing,
        'quality_threshold': quality_threshold,
        'max_workers': max_workers,
        'coverage_config': {
            "min_overlap_lines": 1,
            "min_coverage_ratio": 0.5,
            "max_overage_ratio": 100,
            "allow_partial_coverage": True,
            "require_exact_boundaries": False
        }
    }
    
    # 处理聚类
    print(f"Processing {len(clusters) if cluster_limit == 0 else min(cluster_limit, len(clusters))} clusters...")
    sys.stdout.flush()
    
    cluster_results = process_clusters_parallel(clusters, config)
    
    # 生成汇总报告
    global_summary = generate_summary_reports(cluster_results, config)
    
    # 打印统计信息
    print_statistics(global_summary)
    
    return global_summary

def main():
    """主函数"""
    INPUT_JSON_PATH = "/home/zyw/llm_on_code/llm_on_code_optimization/python/2-cluster_new/result_100/test_2.json"
    KNOWLEDGE_BASE_PATH = "/home/zyw/llm_on_code/llm_on_code_optimization/knowledge_base/"
    ROOT_PATH = "/home/zyw/llm_on_code/llm_on_code_optimization/"
    CLUSTER_LIMIT = 0  # 处理前n个聚类，0表示全部
    REUSE_EXISTING = True
    QUALITY_THRESHOLD = 0.4  # 40%
    MAX_WORKERS = 16
    
    # 输出运行配置
    print(f"{'='*60}")
    print(f"TEST_CLUSTER.PY - CONFIGURATION")
    print(f"{'='*60}")
    print(f"Input JSON: {INPUT_JSON_PATH}")
    print(f"Knowledge base: {KNOWLEDGE_BASE_PATH}")
    print(f"Root path: {ROOT_PATH}")
    print(f"Cluster limit: {CLUSTER_LIMIT if CLUSTER_LIMIT > 0 else 'All'}")
    print(f"Reuse existing: {REUSE_EXISTING}")
    print(f"Quality threshold: {QUALITY_THRESHOLD*100}%")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"{'='*60}")
    print()
    sys.stdout.flush()
    
    test_cluster_file(
        input_json_path=INPUT_JSON_PATH,
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        root_path=ROOT_PATH,
        cluster_limit=CLUSTER_LIMIT,
        reuse_existing=REUSE_EXISTING,
        quality_threshold=QUALITY_THRESHOLD,
        max_workers=MAX_WORKERS
    )

if __name__ == "__main__":
    main()