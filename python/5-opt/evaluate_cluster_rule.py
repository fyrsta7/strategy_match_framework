import os
import json
import glob
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from import_configs import global_config, opt_config
from get_semgrep_result_commit import process_all_semgrep_rules

# 全局变量
CLUSTER_FILE_NAME = "0_85_5_False_order"
# 每个cluster中使用多少个commit对应的semgrep rule，在json文件中从前往后查找每个commit
# 设置为-1表示考虑所有commit
COMMIT_NUM_PER_CLUSTER = 10
CLUSTER_JSON_PATH = opt_config.root_path + f"python/2-cluster_new/result_30342/{CLUSTER_FILE_NAME}.json"
TEST_JSON_PATH = opt_config.eval1_path + "benchmark/151_human.json"
MAX_RULES_PER_COMMIT = opt_config.EVAL_MAX_RULES_PER_COMMIT
MAX_WORKERS_COMMITS = opt_config.EVAL_MAX_WORKERS_COMMITS
MAX_WORKERS_RULES = opt_config.EVAL_MAX_WORKERS_RULES
REUSE_TEMP_RESULTS = opt_config.EVAL_REUSE_TEMP_RESULTS
OUTPUT_BASE_DIR = opt_config.root_path + f"semgrep_result/{CLUSTER_FILE_NAME}_commit{COMMIT_NUM_PER_CLUSTER}/"
TEMP_BASE_DIR = opt_config.root_path + "semgrep/temp_eval_test/"
SEMGREP_KNOWLEDGE_BASE_DIR = opt_config.root_path + "knowledge_base_all/"
COMMIT_KNOWLEDGE_BASE_DIR = opt_config.eval1_path + "knowledge_base/"

def print_info(message):
    """关键信息输出 - 始终显示"""
    print(message, flush=True)

def print_verbose(message):
    """详细信息输出 - 仅在VERBOSE模式下显示"""
    if opt_config.VERBOSE:
        print(message, flush=True)

class ThreadSafeCounter:
    """线程安全的计数器"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self, amount=1):
        with self._lock:
            self._value += amount
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

def load_cluster_data():
    """加载聚类JSON数据"""
    try:
        with open(CLUSTER_JSON_PATH, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        
        if 'clusters' not in cluster_data:
            raise ValueError("No 'clusters' field found in cluster JSON")
        
        return cluster_data['clusters']
    except Exception as e:
        print_info(f"Error loading cluster data: {e}")
        return None

def load_test_data():
    """加载测试集JSON数据"""
    try:
        with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        return test_data
    except Exception as e:
        print_info(f"Error loading test data: {e}")
        return None

def validate_commit_num_per_cluster():
    """验证COMMIT_NUM_PER_CLUSTER参数的合理性"""
    if COMMIT_NUM_PER_CLUSTER == -1:
        print_info("COMMIT_NUM_PER_CLUSTER set to -1: will use all available commits per cluster")
        return True
    elif COMMIT_NUM_PER_CLUSTER > 0:
        print_info(f"COMMIT_NUM_PER_CLUSTER set to {COMMIT_NUM_PER_CLUSTER}: will use up to {COMMIT_NUM_PER_CLUSTER} commits per cluster")
        return True
    else:
        print_info(f"Warning: Invalid COMMIT_NUM_PER_CLUSTER value {COMMIT_NUM_PER_CLUSTER}, using default value 10")
        return False

def scan_available_rules(clusters):
    """扫描聚类中可用的Semgrep规则，按cluster限制commit数量"""
    available_rules = []
    cluster_stats = {}
    
    # 验证参数
    use_limit = COMMIT_NUM_PER_CLUSTER != -1
    limit_value = COMMIT_NUM_PER_CLUSTER if use_limit else float('inf')
    
    for cluster in clusters:
        cluster_id = cluster.get('cluster_id', 'unknown')
        print_verbose(f"Scanning rules for cluster: {cluster_id}")
        
        cluster_commit_count = 0
        cluster_total_commits = len(cluster.get('commits', []))
        cluster_skipped_commits = 0
        
        # 按JSON文件中的顺序遍历commits
        for commit in cluster.get('commits', []):
            # 如果已达到该cluster的commit数量限制，跳过剩余commits
            if use_limit and cluster_commit_count >= limit_value:
                cluster_skipped_commits = cluster_total_commits - cluster_commit_count
                break
            
            repo_name = commit.get('repository_name')
            commit_hash = commit.get('hash')
            
            if not repo_name or not commit_hash:
                continue
            
            # 构建semgrep规则目录路径
            semgrep_dir = os.path.join(
                SEMGREP_KNOWLEDGE_BASE_DIR, repo_name, 
                'modified_file', commit_hash, 'semgrep'
            )
            
            if not os.path.exists(semgrep_dir):
                continue
            
            # 查找YAML规则文件
            yaml_files = glob.glob(os.path.join(semgrep_dir, "*.yaml"))
            yaml_files.extend(glob.glob(os.path.join(semgrep_dir, "*.yml")))
            
            if yaml_files:
                available_rules.append({
                    'cluster_id': cluster_id,
                    'commit': commit,
                    'semgrep_dir': semgrep_dir,
                    'rule_count': len(yaml_files)
                })
                cluster_commit_count += 1
                print_verbose(f"Found {len(yaml_files)} rules for {repo_name}:{commit_hash[:8]} (cluster {cluster_id}, #{cluster_commit_count})")
        
        # 记录cluster统计信息
        cluster_stats[cluster_id] = {
            'total_commits': cluster_total_commits,
            'used_commits': cluster_commit_count,
            'skipped_commits': cluster_skipped_commits
        }
        
        if cluster_commit_count > 0:
            print_verbose(f"Cluster {cluster_id}: used {cluster_commit_count}/{cluster_total_commits} commits")
        else:
            print_verbose(f"Cluster {cluster_id}: no available rules found in any commits")
    
    # 输出总体统计
    total_used_commits = sum(stats['used_commits'] for stats in cluster_stats.values())
    total_skipped_commits = sum(stats['skipped_commits'] for stats in cluster_stats.values())
    
    print_info(f"Total available rule commits: {len(available_rules)}")
    print_info(f"Commits used: {total_used_commits}, skipped due to limit: {total_skipped_commits}")
    
    # 按cluster显示详细统计
    if opt_config.VERBOSE:
        print_verbose("\nCluster-wise commit usage:")
        for cluster_id, stats in cluster_stats.items():
            if stats['used_commits'] > 0:
                print_verbose(f"  Cluster {cluster_id}: {stats['used_commits']}/{stats['total_commits']} commits used")
    
    return available_rules

def find_before_file(commit_dir):
    """查找before文件"""
    pattern = os.path.join(commit_dir, "before.*")
    matches = glob.glob(pattern)
    
    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches[0]
    else:
        raise ValueError(f"Multiple before files found: {matches}")

def validate_test_commit(test_commit):
    """验证测试commit的before文件存在性"""
    repo_name = test_commit.get('repository_name')
    commit_hash = test_commit.get('hash')
    
    if not repo_name or not commit_hash:
        return False, "Missing repository_name or hash"
    
    commit_dir = os.path.join(
        COMMIT_KNOWLEDGE_BASE_DIR, repo_name, 
        'modified_file', commit_hash
    )
    
    if not os.path.exists(commit_dir):
        return False, f"Commit directory not found: {commit_dir}"
    
    before_file = find_before_file(commit_dir)
    if not before_file:
        return False, f"before file not found in {commit_dir}"
    
    return True, before_file

def validate_code_snippet_in_func(line_start, line_end, line_offset, before_func_total_lines):
    """验证代码片段是否属于before_func文件"""
    # 计算在before_func中的行号
    func_line_start = line_start - line_offset
    func_line_end = line_end - line_offset
    
    # 检查是否完全在before_func范围内
    if 1 <= func_line_start <= func_line_end <= before_func_total_lines:
        return True, func_line_start, func_line_end
    else:
        return False, None, None

def generate_temp_hash(test_commit, rule_commit):
    """基于两个commit信息生成唯一哈希值"""
    content = f"{test_commit['repository_name']}:{test_commit['hash']}:{rule_commit['repository_name']}:{rule_commit['hash']}"
    return hashlib.md5(content.encode()).hexdigest()[:16]

def build_temp_dir_path(temp_hash):
    """构建临时目录路径"""
    return os.path.join(TEMP_BASE_DIR, temp_hash)

def build_output_path(test_commit):
    """构建输出文件路径"""
    repo_name = test_commit['repository_name']
    commit_hash = test_commit['hash']
    output_dir = os.path.join(OUTPUT_BASE_DIR, repo_name)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{commit_hash}.json")

def run_single_rule_set(test_commit, rule_commit_info, before_file):
    """运行单个commit的规则集"""
    rule_commit = rule_commit_info['commit']
    semgrep_dir = rule_commit_info['semgrep_dir']
    
    # 生成临时目录
    temp_hash = generate_temp_hash(test_commit, rule_commit)
    temp_dir = build_temp_dir_path(temp_hash)
    
    # 检查是否复用现有结果
    if REUSE_TEMP_RESULTS and os.path.exists(temp_dir):
        yaml_files = glob.glob(os.path.join(semgrep_dir, "*.yaml"))
        yaml_files.extend(glob.glob(os.path.join(semgrep_dir, "*.yml")))
        expected_files = min(len(yaml_files), MAX_RULES_PER_COMMIT)
        
        existing_files = glob.glob(os.path.join(temp_dir, "*.json"))
        if len(existing_files) >= expected_files:
            print_verbose(f"Reusing existing results in {temp_dir}")
            return temp_dir, len(existing_files)
    
    # 执行规则
    try:
        total_count, success_count, failed_count, skipped_count, failed_files = process_all_semgrep_rules(
            yaml_dir_path=semgrep_dir,
            target_file_path=before_file,
            json_output_dir=temp_dir,
            skip_existing=REUSE_TEMP_RESULTS
        )
        
        return temp_dir, success_count
    except Exception as e:
        print_verbose(f"Error running rules: {e}")
        return None, 0

def extract_core_findings(json_file_path, test_commit_info):
    """从Semgrep JSON结果中提取核心信息并进行before_func归属判断"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        line_offset = test_commit_info.get('line_offset', 0)
        before_func_total_lines = test_commit_info.get('before_func_total_lines', 0)
        
        findings = []
        for finding in result.get('results', []):
            line_start = finding.get('start', {}).get('line', 0)
            line_end = finding.get('end', {}).get('line', 0)
            
            # 验证是否属于before_func
            belongs_to_func, func_line_start, func_line_end = validate_code_snippet_in_func(
                line_start, line_end, line_offset, before_func_total_lines
            )
            
            core_info = {
                'rule_id': finding.get('check_id', 'unknown'),
                'message': finding.get('extra', {}).get('message', ''),
                'line_start': line_start,
                'line_end': line_end,
                'severity': finding.get('extra', {}).get('severity', 'unknown'),
                'in_before_file_lines': {'start': line_start, 'end': line_end},
                'belongs_to_func': belongs_to_func,
                'in_func_file_lines': {
                    'start': func_line_start, 
                    'end': func_line_end
                } if belongs_to_func else None
            }
            findings.append(core_info)
        
        return findings
    except Exception as e:
        print_verbose(f"Error extracting findings from {json_file_path}: {e}")
        return []

def aggregate_rule_results(temp_dir, rule_commit_info, test_commit_info):
    """聚合单个commit规则集的结果"""
    if not temp_dir or not os.path.exists(temp_dir):
        return {
            'commit_info': rule_commit_info['commit'],
            'cluster_id': rule_commit_info['cluster_id'],
            'rules_used': 0,
            'total_findings_count': 0,
            'func_related_findings_count': 0,
            'func_coverage_ratio': 0.0,
            'findings_detail': []
        }
    
    json_files = glob.glob(os.path.join(temp_dir, "*.json"))
    json_files = sorted(json_files)[:MAX_RULES_PER_COMMIT]
    
    all_findings = []
    func_related_count = 0
    
    for json_file in json_files:
        rule_filename = os.path.basename(json_file)
        findings = extract_core_findings(json_file, test_commit_info)
        
        for finding in findings:
            finding['rule_file'] = rule_filename
            all_findings.append(finding)
            if finding['belongs_to_func']:
                func_related_count += 1
    
    total_findings_count = len(all_findings)
    func_coverage_ratio = func_related_count / total_findings_count if total_findings_count > 0 else 0.0
    
    return {
        'commit_info': rule_commit_info['commit'],
        'cluster_id': rule_commit_info['cluster_id'],
        'rules_used': len(json_files),
        'total_findings_count': total_findings_count,
        'func_related_findings_count': func_related_count,
        'func_coverage_ratio': func_coverage_ratio,
        'findings_detail': all_findings
    }

def execute_rules_on_commit(test_commit, available_rules):
    """对单个测试commit执行所有规则"""
    # 验证测试commit
    valid, before_file = validate_test_commit(test_commit)
    if not valid:
        print_verbose(f"Invalid test commit {test_commit.get('hash', 'unknown')[:8]}: {before_file}")
        return None
    
    print_verbose(f"Processing test commit: {test_commit['repository_name']}:{test_commit['hash'][:8]}")
    
    # 并行执行规则
    cluster_results = {}
    rules_executed_counter = ThreadSafeCounter()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_RULES) as executor:
        future_to_rule = {}
        
        for rule_commit_info in available_rules:
            future = executor.submit(run_single_rule_set, test_commit, rule_commit_info, before_file)
            future_to_rule[future] = rule_commit_info
        
        for future in as_completed(future_to_rule):
            rule_commit_info = future_to_rule[future]
            try:
                temp_dir, rules_count = future.result()
                rules_executed_counter.increment(rules_count)
                
                # 聚合结果
                aggregated_result = aggregate_rule_results(temp_dir, rule_commit_info, test_commit)
                
                cluster_id = rule_commit_info['cluster_id']
                if cluster_id not in cluster_results:
                    cluster_results[cluster_id] = []
                
                cluster_results[cluster_id].append(aggregated_result)
                
            except Exception as e:
                print_verbose(f"Error processing rule commit: {e}")
    
    # 构建最终结果
    commit_summary = build_commit_summary(test_commit, cluster_results, rules_executed_counter.value)
    return commit_summary

def build_commit_summary(test_commit, cluster_results, total_rules_executed):
    """构建单个测试commit的完整结果摘要"""
    total_findings = sum(
        sum(result['total_findings_count'] for result in cluster_commits)
        for cluster_commits in cluster_results.values()
    )
    
    total_func_related_findings = sum(
        sum(result['func_related_findings_count'] for result in cluster_commits)
        for cluster_commits in cluster_results.values()
    )
    
    overall_func_coverage_ratio = total_func_related_findings / total_findings if total_findings > 0 else 0.0
    
    return {
        'test_commit_info': {
            'repository_name': test_commit['repository_name'],
            'hash': test_commit['hash'],
            'file_path': test_commit.get('file_path', ''),
            'line_offset': test_commit.get('line_offset', 0),
            'before_func_total_lines': test_commit.get('before_func_total_lines', 0)
        },
        'execution_summary': {
            'total_rule_commits': sum(len(cluster_commits) for cluster_commits in cluster_results.values()),
            'total_rules_executed': total_rules_executed,
            'total_findings': total_findings,
            'total_func_related_findings': total_func_related_findings,
            'overall_func_coverage_ratio': overall_func_coverage_ratio,
            'execution_time': time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'cluster_results': [
            {
                'cluster_id': cluster_id,
                'rule_commits': cluster_commits
            }
            for cluster_id, cluster_commits in cluster_results.items()
        ]
    }

def save_commit_results(commit_summary, output_path):
    """保存结果到JSON文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(commit_summary, f, indent=opt_config.DEFAULT_OUTPUT_INDENT, ensure_ascii=False)
        return True
    except Exception as e:
        print_verbose(f"Error saving results to {output_path}: {e}")
        return False

def process_single_test_commit(test_commit, available_rules):
    """处理单个测试commit"""
    try:
        commit_summary = execute_rules_on_commit(test_commit, available_rules)
        if not commit_summary:
            return False, f"Failed to process commit {test_commit.get('hash', 'unknown')}"
        
        output_path = build_output_path(test_commit)
        success = save_commit_results(commit_summary, output_path)
        
        if success:
            return True, commit_summary['execution_summary']
        else:
            return False, "Failed to save results"
            
    except Exception as e:
        return False, str(e)

def main():
    """主函数"""
    print_info("=== Cluster Rules Evaluation Started ===")
    
    # 验证参数
    validate_commit_num_per_cluster()
    
    # 加载数据
    print_info("Loading input data...")
    clusters = load_cluster_data()
    test_data = load_test_data()
    
    if not clusters or not test_data:
        print_info("Failed to load input data")
        return
    
    # 扫描可用规则
    print_info("Scanning available rules...")
    available_rules = scan_available_rules(clusters)
    
    if not available_rules:
        print_info("No available rules found")
        return
    
    print_info(f"Found {len(available_rules)} rule commits")
    print_info(f"Test commits to process: {len(test_data)}")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(TEMP_BASE_DIR, exist_ok=True)
    
    # 并行处理测试commits
    success_count = 0
    total_rules_executed = 0
    total_findings = 0
    total_func_related_findings = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_COMMITS) as executor:
        future_to_commit = {}
        
        for test_commit in test_data:
            future = executor.submit(process_single_test_commit, test_commit, available_rules)
            future_to_commit[future] = test_commit
        
        # 使用tqdm显示进度
        with tqdm(total=len(test_data), desc="Processing test commits", unit="commit") as pbar:
            for future in as_completed(future_to_commit):
                test_commit = future_to_commit[future]
                try:
                    success, result = future.result()
                    if success:
                        success_count += 1
                        total_rules_executed += result['total_rules_executed']
                        total_findings += result['total_findings']
                        total_func_related_findings += result['total_func_related_findings']
                        pbar.set_postfix({
                            'success': success_count,
                            'rules': total_rules_executed,
                            'findings': total_findings,
                            'func_findings': total_func_related_findings
                        })
                    else:
                        print_verbose(f"Failed to process {test_commit.get('hash', 'unknown')[:8]}: {result}")
                except Exception as e:
                    print_verbose(f"Exception processing {test_commit.get('hash', 'unknown')[:8]}: {e}")
                finally:
                    pbar.update(1)
    
    # 输出最终统计
    overall_func_coverage = total_func_related_findings / total_findings if total_findings > 0 else 0.0
    print_info(f"\n=== Final Statistics ===")
    print_info(f"Test commits processed: {success_count}/{len(test_data)}")
    print_info(f"Total rules executed: {total_rules_executed}")
    print_info(f"Total findings: {total_findings}")
    print_info(f"Total func-related findings: {total_func_related_findings}")
    print_info(f"Overall func coverage ratio: {overall_func_coverage:.2%}")
    print_info(f"Results saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()