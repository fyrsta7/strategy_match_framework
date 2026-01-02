import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from import_configs import global_config, opt_config

# 全局变量 - 测试配置
TEST_JSON_PATH = global_config.root_path + "python/5-opt/eval1_benchmark/test_3_commits.json"
RESULT_BASE_DIR = global_config.root_path + "semgrep_result/TEST_0_85_5_False_order_commit3/"
IGNORE_SELF_RULES = False  # 不忽略自身commit的规则结果（随机函数优化场景）
REUSE_EXISTING_SUMMARY = False  # 是否复用已存在的摘要文件
MAX_WORKERS = 4  # 减少并行数用于测试
VALIDATE_FUNC_CONSISTENCY = True  # 是否验证func字段一致性(调试用)

# 输出文件后缀
CLUSTER_SUFFIX = "_summary_cluster.json"  # 聚类内汇总文件后缀

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

def load_test_commits():
    """加载测试集commit列表"""
    try:
        with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print_info(f"Loaded {len(test_data)} test commits")
        return test_data
    except Exception as e:
        print_info(f"Error loading test data: {e}")
        return None

def load_commit_results(commit):
    """加载单个commit的检测结果JSON"""
    repo_name = commit['repository_name']
    commit_hash = commit['hash']
    
    result_path = os.path.join(RESULT_BASE_DIR, repo_name, f"{commit_hash}.json")
    
    if not os.path.exists(result_path):
        return None, f"Result file not found: {result_path}"
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        return result_data, None
    except Exception as e:
        return None, f"Error loading result file: {e}"

def validate_result_file(result_path):
    """验证结果文件存在性和有效性"""
    if not os.path.exists(result_path):
        return False, "File not found"
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, None
    except Exception as e:
        return False, f"Invalid JSON: {e}"

def should_ignore_commit_rules(rule_commit, test_commit):
    """判断是否应忽略某个规则commit的结果"""
    if not IGNORE_SELF_RULES:
        return False
    
    return (rule_commit['repository_name'] == test_commit['repository_name'] and 
            rule_commit['hash'] == test_commit['hash'])

def parse_code_segments(findings_detail, rule_commit_info):
    """解析检测结果中的代码片段信息,只处理属于func的findings"""
    segments = []
    
    # 提取commit来源信息
    source_commit = {
        'repository_name': rule_commit_info.get('repository_name', 'unknown'),
        'hash': rule_commit_info.get('hash', 'unknown')
    }
    
    for finding in findings_detail:
        # 只处理属于func的findings
        if not finding.get('belongs_to_func', False):
            continue
        
        # 检查必需字段
        in_func_lines = finding.get('in_func_file_lines')
        in_before_lines = finding.get('in_before_file_lines')
        
        if not in_func_lines or not in_before_lines:
            continue
        
        func_start = in_func_lines.get('start', 0)
        func_end = in_func_lines.get('end', 0)
        
        if func_start > 0 and func_end > 0:
            segment = {
                'line_start': func_start,  # 使用func文件中的行号
                'line_end': func_end,
                'rule_id': finding.get('rule_id', 'unknown'),
                'message': finding.get('message', ''),
                'severity': finding.get('severity', 'unknown'),
                'source_commit': source_commit,
                # 保留原始定位信息
                'in_func_file_lines': in_func_lines,
                'in_before_file_lines': in_before_lines,
                'belongs_to_func': True
            }
            segments.append(segment)
    
    return segments

def extract_cluster_findings(cluster_result, test_commit):
    """提取聚类中的所有检测结果"""
    all_segments = []
    
    for rule_commit_result in cluster_result.get('rule_commits', []):
        rule_commit = rule_commit_result.get('commit_info', {})
        
        # 检查是否应忽略此commit的规则
        if should_ignore_commit_rules(rule_commit, test_commit):
            print_verbose(f"Ignoring self rules from {rule_commit.get('repository_name', 'unknown')}:{rule_commit.get('hash', 'unknown')[:8]}")
            continue
        
        findings_detail = rule_commit_result.get('findings_detail', [])
        segments = parse_code_segments(findings_detail, rule_commit)
        all_segments.extend(segments)
    
    return all_segments

def sort_segments_by_detection_count(segments):
    """按检测次数降序、行号升序排序代码片段"""
    return sorted(segments, key=lambda x: (-x['detection_count'], x['line_start'], x['line_end']))

def reassign_segment_ids(segments, start_id=1):
    """重新为排序后的片段分配递增ID"""
    for i, segment in enumerate(segments):
        segment['id'] = start_id + i
    return segments

def deduplicate_segments_with_ids(cluster_segments):
    """代码片段去重,基于func文件行号,保留必要的原始信息"""
    segment_map = {}
    severity_priority = {'error': 3, 'warning': 2, 'info': 1, 'unknown': 0}
    
    for segment in cluster_segments:
        # 使用func文件中的行号作为去重键
        key = (segment['line_start'], segment['line_end'])
        
        if key not in segment_map:
            # 第一次遇到这个行号范围,保留所有信息
            segment_map[key] = {
                'line_start': segment['line_start'],
                'line_end': segment['line_end'],
                'detection_count': 0,
                'rule_details': {},  # 新的统一存储结构: rule_id -> {message, severity, source_commit}
                # 保留原始定位信息(从第一个finding中获取)
                'in_func_file_lines': segment['in_func_file_lines'],
                'in_before_file_lines': segment['in_before_file_lines'],
                'belongs_to_func': segment['belongs_to_func']
            }
        
        segment_info = segment_map[key]
        
        # 可选的一致性验证
        if VALIDATE_FUNC_CONSISTENCY:
            if (segment_info['in_before_file_lines'] != segment['in_before_file_lines'] or
                segment_info['in_func_file_lines'] != segment['in_func_file_lines']):
                print_verbose(f"Warning: Inconsistent line ranges found for key {key}")
        
        # 新的聚合逻辑：基于rule_id去重
        rule_id = segment['rule_id']
        if rule_id not in segment_info['rule_details']:
            # 新规则：记录所有信息
            segment_info['rule_details'][rule_id] = {
                'message': segment['message'],
                'severity': segment['severity'],
                'source_commit': segment['source_commit']
            }
            segment_info['detection_count'] += 1
        else:
            # 已存在规则：可选择性更新severity（选择更高优先级）
            existing = segment_info['rule_details'][rule_id]
            if severity_priority.get(segment['severity'], 0) > severity_priority.get(existing['severity'], 0):
                existing['severity'] = segment['severity']
    
    # 转换为最终格式
    unique_segments = []
    
    for segment_info in segment_map.values():
        # 确保一一对应：按rule_id排序
        rule_items = sorted(segment_info['rule_details'].items())
        
        # 从所有rule_details中选择最高优先级的severity
        highest_severity = max(
            (detail['severity'] for detail in segment_info['rule_details'].values()),
            key=lambda x: severity_priority.get(x, 0)
        ) if segment_info['rule_details'] else 'unknown'
        
        unique_segment = {
            'line_start': segment_info['line_start'],
            'line_end': segment_info['line_end'],
            'detection_count': len(rule_items),  # 等于唯一rule_id数量
            'rule_ids': [item[0] for item in rule_items],
            'messages': [item[1]['message'] for item in rule_items if item[1]['message']],  # 过滤空消息
            'severity': highest_severity,
            'source_commits': [item[1]['source_commit'] for item in rule_items],
            # 保留的原始定位信息
            'in_func_file_lines': segment_info['in_func_file_lines'],
            'in_before_file_lines': segment_info['in_before_file_lines'],
            'belongs_to_func': segment_info['belongs_to_func']
        }
        
        # 数据一致性验证
        if opt_config.VERBOSE:
            assert len(unique_segment['rule_ids']) == len(unique_segment['source_commits']) == unique_segment['detection_count']
            assert len(unique_segment['messages']) <= len(unique_segment['rule_ids'])
            print_verbose(f"Segment ({unique_segment['line_start']}, {unique_segment['line_end']}): "
                         f"detection_count={unique_segment['detection_count']}, "
                         f"rule_count={len(unique_segment['rule_ids'])}, "
                         f"message_count={len(unique_segment['messages'])}")
        
        unique_segments.append(unique_segment)
    
    return unique_segments

def aggregate_cluster_results(cluster_data, test_commit):
    """汇总单个聚类的结果（不排序，不分配ID）"""
    cluster_id = cluster_data.get('cluster_id', 'unknown')
    
    # 提取所有代码片段
    all_segments = extract_cluster_findings(cluster_data, test_commit)
    
    # 统计过滤前的信息
    total_findings_before_filter = 0
    for rule_commit_result in cluster_data.get('rule_commits', []):
        rule_commit = rule_commit_result.get('commit_info', {})
        if not should_ignore_commit_rules(rule_commit, test_commit):
            total_findings_before_filter += rule_commit_result.get('total_findings_count', 0)
    
    func_related_findings = len(all_segments)
    
    # 去重（聚类内去重合并）
    unique_segments = deduplicate_segments_with_ids(all_segments)
    
    # 注意：不在这里排序和分配ID，留待全局处理
    
    return {
        'cluster_id': cluster_id,
        'total_findings_before_filter': total_findings_before_filter,
        'func_related_findings': func_related_findings,
        'unique_segments': unique_segments
    }

def calculate_cross_cluster_detection(aggregated_clusters):
    """计算跨聚类检测统计：相同行号范围在多少个聚类中被检测到"""
    line_range_to_clusters = {}
    
    # 收集每个行号范围出现在哪些聚类中
    for cluster_summary in aggregated_clusters:
        cluster_id = cluster_summary['cluster_id']
        for segment in cluster_summary['unique_segments']:
            key = (segment['line_start'], segment['line_end'])
            if key not in line_range_to_clusters:
                line_range_to_clusters[key] = set()
            line_range_to_clusters[key].add(cluster_id)
    
    # 转换为行号范围 → 聚类数量的映射
    cross_cluster_stats = {}
    for key, cluster_set in line_range_to_clusters.items():
        cross_cluster_stats[key] = len(cluster_set)
    
    return cross_cluster_stats

def collect_all_segments_with_cluster_info(aggregated_clusters, cross_cluster_stats):
    """收集所有片段并添加聚类信息和跨聚类统计"""
    all_segments = []
    
    for cluster_summary in aggregated_clusters:
        cluster_id = cluster_summary['cluster_id']
        
        for segment in cluster_summary['unique_segments']:
            # 复制原始片段信息
            segment_with_cluster_info = segment.copy()
            
            # 添加聚类来源信息
            segment_with_cluster_info['source_cluster_id'] = cluster_id
            
            # 添加跨聚类检测统计
            key = (segment['line_start'], segment['line_end'])
            segment_with_cluster_info['cross_cluster_detection'] = cross_cluster_stats.get(key, 1)
            
            all_segments.append(segment_with_cluster_info)
    
    return all_segments

def sort_all_segments_globally(all_segments):
    """对所有片段进行全局排序"""
    return sorted(all_segments, key=lambda x: (-x['detection_count'], x['line_start'], x['line_end']))

def process_cluster_mode(original_result, test_commit):
    """处理聚类内汇总模式（全局排序）"""
    # 阶段1：各聚类独立汇总
    aggregated_clusters = []
    total_findings_before_filter = 0
    total_func_related_findings = 0
    
    for cluster_data in original_result.get('cluster_results', []):
        cluster_summary = aggregate_cluster_results(cluster_data, test_commit)
        aggregated_clusters.append(cluster_summary)
        total_findings_before_filter += cluster_summary['total_findings_before_filter']
        total_func_related_findings += cluster_summary['func_related_findings']
    
    # 阶段2：计算跨聚类统计
    cross_cluster_stats = calculate_cross_cluster_detection(aggregated_clusters)
    
    # 阶段3：收集所有片段并添加元信息
    all_segments = collect_all_segments_with_cluster_info(aggregated_clusters, cross_cluster_stats)
    
    # 阶段4：全局排序
    all_segments_sorted = sort_all_segments_globally(all_segments)
    
    # 阶段5：全局ID分配
    all_segments_sorted = reassign_segment_ids(all_segments_sorted, 1)
    
    func_coverage_ratio = total_func_related_findings / total_findings_before_filter if total_findings_before_filter > 0 else 0.0
    
    summary = {
        'test_commit_info': original_result.get('test_commit_info', {}),
        'execution_summary': {
            'total_rule_commits': original_result.get('execution_summary', {}).get('total_rule_commits', 0),
            'total_rules_executed': original_result.get('execution_summary', {}).get('total_rules_executed', 0),
            'total_findings_before_filter': total_findings_before_filter,
            'total_func_related_findings': total_func_related_findings,
            'func_coverage_ratio': func_coverage_ratio,
            'total_unique_segments_global': len(all_segments_sorted),
            'processing_mode': 'cluster_aggregated_global_sorted',
            'ignored_self_rules': IGNORE_SELF_RULES,
            'summary_time': time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'global_segments': all_segments_sorted,
        'cluster_statistics': [
            {
                'cluster_id': cluster['cluster_id'],
                'segments_count': len(cluster['unique_segments']),
                'total_findings_before_filter': cluster['total_findings_before_filter'],
                'func_related_findings': cluster['func_related_findings']
            }
            for cluster in aggregated_clusters
        ]
    }
    
    return summary

def build_summary_path(commit):
    """构建聚类摘要文件路径"""
    repo_name = commit['repository_name']
    commit_hash = commit['hash']
    
    summary_dir = os.path.join(RESULT_BASE_DIR, repo_name)
    os.makedirs(summary_dir, exist_ok=True)
    
    return os.path.join(summary_dir, f"{commit_hash}{CLUSTER_SUFFIX}")

def save_summary_results(summary_data, output_path):
    """保存摘要结果"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=opt_config.DEFAULT_OUTPUT_INDENT, ensure_ascii=False)
        return True
    except Exception as e:
        print_verbose(f"Error saving summary to {output_path}: {e}")
        return False

def process_single_commit(commit):
    """处理单个commit的聚类摘要生成"""
    repo_name = commit.get('repository_name', 'unknown')
    commit_hash = commit.get('hash', 'unknown')
    
    try:
        # 加载原始检测结果
        original_result, error_msg = load_commit_results(commit)
        if not original_result:
            return {'success': False, 'message': f"Failed to load results: {error_msg}"}
        
        # 处理聚类模式
        cluster_path = build_summary_path(commit)
        
        # 检查是否复用
        if REUSE_EXISTING_SUMMARY and os.path.exists(cluster_path):
            valid, _ = validate_result_file(cluster_path)
            if valid:
                return {'success': True, 'message': 'Reused existing summary'}
        
        # 生成新的摘要
        cluster_summary = process_cluster_mode(original_result, commit)
        success = save_summary_results(cluster_summary, cluster_path)
        if success:
            total_segments = cluster_summary['execution_summary']['total_unique_segments_global']
            return {'success': True, 'message': f'Generated with {total_segments} segments'}
        else:
            return {'success': False, 'message': 'Failed to save cluster summary'}
            
    except Exception as e:
        return {'success': False, 'message': f"Exception: {str(e)}"}

def main():
    """主函数"""
    print_info("=== Cluster Results Summarization Started (TEST MODE) ===")
    
    # 加载测试commit列表
    print_info("Loading test commits...")
    test_commits = load_test_commits()
    
    if not test_commits:
        print_info("Failed to load test commits")
        return
    
    print_info(f"Processing {len(test_commits)} test commits")
    print_info(f"Ignore self rules: {IGNORE_SELF_RULES}")
    print_info(f"Reuse existing summaries: {REUSE_EXISTING_SUMMARY}")
    print_info(f"Validate func consistency: {VALIDATE_FUNC_CONSISTENCY}")
    
    # 统计变量
    success_counter = ThreadSafeCounter()
    reused_counter = ThreadSafeCounter()
    failed_counter = ThreadSafeCounter()
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_commit = {}
        
        for commit in test_commits:
            future = executor.submit(process_single_commit, commit)
            future_to_commit[future] = commit
        
        # 使用tqdm显示进度
        with tqdm(total=len(test_commits), desc="Summarizing cluster results", unit="commit") as pbar:
            for future in as_completed(future_to_commit):
                commit = future_to_commit[future]
                commit_id = f"{commit.get('repository_name', 'unknown')}:{commit.get('hash', 'unknown')[:8]}"
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        success_counter.increment()
                        if "Reused" in result['message']:
                            reused_counter.increment()
                        print_verbose(f"{commit_id}: {result['message']}")
                    else:
                        failed_counter.increment()
                        print_verbose(f"Failed for {commit_id}: {result['message']}")
                    
                    pbar.set_postfix({
                        'success': success_counter.value,
                        'failed': failed_counter.value
                    })
                    
                except Exception as e:
                    failed_counter.increment()
                    print_verbose(f"Exception processing {commit_id}: {e}")
                finally:
                    pbar.update(1)
    
    # 输出最终统计
    print_info(f"\n=== Final Statistics ===")
    print_info(f"Total commits: {len(test_commits)}")
    print_info(f"Cluster mode - Success: {success_counter.value}, Reused: {reused_counter.value}")
    print_info(f"Failed: {failed_counter.value}")
    print_info(f"Results saved to: {RESULT_BASE_DIR}")

if __name__ == "__main__":
    main()

