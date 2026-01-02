import os
import json
import time
import threading
import contextlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from import_configs import global_config, generate_config
from process_commit import process_commit_semgrep_rules

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

@contextlib.contextmanager
def suppress_stdout_stderr():
    """抑制stdout和stderr输出"""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def find_before_file(commit_dir):
    """查找before文件（任意后缀）"""
    import glob
    pattern = os.path.join(commit_dir, "before.*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]  # 返回第一个匹配的文件
    return None

def build_commit_paths(commit_info):
    """构建commit相关的文件路径"""
    knowledge_base_path = os.path.join(global_config.root_path, generate_config.KNOWLEDGE_BASE_RELATIVE_PATH)
    repo_name = commit_info['repository_name']
    commit_hash = commit_info['hash']
    commit_dir = os.path.join(knowledge_base_path, repo_name, 'modified_file', commit_hash)
    
    # 查找before文件
    before_file_path = find_before_file(commit_dir)
    
    return {
        'commit_dir': commit_dir,
        'diff_file': os.path.join(commit_dir, 'diff.txt'),
        'before_file': before_file_path,
        'semgrep_dir': os.path.join(commit_dir, 'semgrep')
    }

def is_commit_processable(commit_info):
    """检查commit是否可以处理"""
    # 检查必要字段
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
        
        # 检查diff文件是否为空
        with open(paths['diff_file'], 'r', encoding='utf-8') as f:
            if not f.read().strip():
                return False, "diff.txt is empty"
        
        return True, None
    except Exception as e:
        return False, f"Error checking files: {str(e)}"

def should_continue_fixing(json_result, error_msg):
    """判断是否需要继续修复迭代（复用process_once.py的逻辑）"""
    # 如果有执行错误，需要继续修复
    if error_msg:
        return True
    
    # 如果没有JSON结果或没有errors，不需要修复
    if not json_result or not json_result.get("errors"):
        return False
    
    # 检查每个错误的级别
    for error in json_result.get("errors", []):
        level = error.get("level", "").lower()
        # 如果有严重错误，需要继续修复
        if level in ["fatal", "error"]:
            return True
    
    # 所有错误都是警告级别或更低，不需要继续修复
    return False

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
            json_path = os.path.join(semgrep_dir, f"{rule_number}.json")
            
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
                        if not content or ('rules:' not in content and 'rule:' not in content):
                            # YAML文件内容无效，不计入有效规则
                            continue
                except:
                    # 读取YAML文件失败，不计入有效规则
                    continue
                
                # YAML文件有效，检查JSON结果文件
                # 如果JSON文件不存在，认为规则还未运行，不计入有效规则
                if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
                    continue
                
                # JSON文件存在，根据配置决定是否检查错误
                if generate_config.COMMIT_REGENERATE_ON_JSON_ERROR:
                    # 检查是否有fatal/error级别的错误
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            json_result = json.load(f)
                        
                        # 如果JSON结果中有fatal/error级别的错误，不计入有效规则
                        if should_continue_fixing(json_result, None):
                            continue
                    except:
                        # JSON文件解析失败，不计入有效规则
                        continue
                
                # YAML和JSON都存在且有效（或配置为不检查JSON错误），计入有效规则
                valid_rules_count += 1
        
        # 只有当规则数量达到COMMIT_GENERATION_COUNT时才认为有足够的规则
        return valid_rules_count >= generate_config.COMMIT_GENERATION_COUNT
    except Exception as e:
        print_verbose(f"Error checking semgrep rules for {commit_info.get('hash', 'unknown')}: {e}")
        return False

def process_single_commit(commit_info, cluster_id, commit_idx, total_commits_in_cluster):
    """处理单个commit的规则生成"""
    commit_hash = commit_info['hash']
    repo_name = commit_info['repository_name']
    
    # 初始化结果结构
    commit_result = {
        "repository_name": repo_name,
        "hash": commit_hash,
        "rules_generated": 0,
        "generation_successful": False,
        "error_message": None
    }
    
    try:
        # 构建路径
        commit_paths = build_commit_paths(commit_info)
        
        if generate_config.SIMPLE_SHOW_COMMIT_PROGRESS:
            print_progress(f"Cluster {cluster_id} - Commit {commit_idx}/{total_commits_in_cluster} "
                          f"({repo_name}:{commit_hash[:8]}) started")
        
        # 生成Semgrep规则
        if generate_config.VERBOSE:
            success_count = process_commit_semgrep_rules(
                commit_dir_path=commit_paths['commit_dir'],
                generation_count=generate_config.COMMIT_GENERATION_COUNT,
                use_langsmith=generate_config.LLM_USE_LANGSMITH,
                reuse_existing=generate_config.COMMIT_REUSE_EXISTING,
                max_round=generate_config.LLM_MAX_GENERATION_ROUNDS,
                regenerate_on_json_error=generate_config.COMMIT_REGENERATE_ON_JSON_ERROR
            )
        else:
            with suppress_stdout_stderr():
                success_count = process_commit_semgrep_rules(
                    commit_dir_path=commit_paths['commit_dir'],
                    generation_count=generate_config.COMMIT_GENERATION_COUNT,
                    use_langsmith=generate_config.LLM_USE_LANGSMITH,
                    reuse_existing=generate_config.COMMIT_REUSE_EXISTING,
                    max_round=generate_config.LLM_MAX_GENERATION_ROUNDS,
                    regenerate_on_json_error=generate_config.COMMIT_REGENERATE_ON_JSON_ERROR
                )
        
        # 更新结果
        commit_result["rules_generated"] = success_count
        commit_result["generation_successful"] = success_count > 0
        
        if generate_config.SIMPLE_SHOW_COMMIT_PROGRESS:
            status = "SUCCESS" if success_count > 0 else "FAILED"
            print_progress(f"Cluster {cluster_id} - Commit {commit_idx}/{total_commits_in_cluster} "
                          f"({repo_name}:{commit_hash[:8]}) completed - {status} ({success_count} rules)")
        
        return True, commit_result
        
    except Exception as e:
        commit_result["error_message"] = str(e)
        if generate_config.SIMPLE_SHOW_COMMIT_PROGRESS:
            print_progress(f"Cluster {cluster_id} - Commit {commit_idx}/{total_commits_in_cluster} "
                          f"({repo_name}:{commit_hash[:8]}) completed - ERROR: {str(e)}")
        return False, commit_result

class ThreadSafeCounter:
    """线程安全的计数器，用于跟踪进度"""
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

def process_single_cluster(cluster_info, cluster_index, total_clusters):
    """处理单个聚类"""
    cluster_id = cluster_info.get('cluster_id', f"cluster_{cluster_index}")
    
    # 初始化聚类结果
    cluster_result = {
        "cluster_id": cluster_id,
        "cluster_index": cluster_index,
        "commits_with_existing_rules": 0,
        "commits_newly_processed": 0,
        "commits_newly_successful": 0,
        "total_commits_with_rules": 0,
        "total_rules_generated": 0,
        "commit_details": []
    }
    
    print_progress(f"Cluster {cluster_id} ({cluster_index + 1}/{total_clusters}) started")
    
    # 筛选可处理的commits
    processable_commits = []
    skipped_count = 0
    
    for commit in cluster_info['commits']:
        processable, error_msg = is_commit_processable(commit)
        if processable:
            processable_commits.append(commit)
        else:
            skipped_count += 1
            print_verbose(f"Cluster {cluster_id} - Skipped commit {commit.get('hash', 'unknown')[:8]}: {error_msg}")
    
    if skipped_count > 0:
        print_verbose(f"Cluster {cluster_id} skipped {skipped_count} unprocessable commits")
    
    if not processable_commits:
        print_progress(f"Cluster {cluster_id} ({cluster_index + 1}/{total_clusters}) completed - No processable commits")
        return cluster_result
    
    # 选择前n个可处理的commits（如果总数量不够n的话就给所有commit生成semgrep规则）
    target_count = generate_config.SIMPLE_COMMITS_PER_CLUSTER
    top_n_commits = processable_commits[:target_count] if len(processable_commits) >= target_count else processable_commits
    
    # 检查这前n个commits中哪些已有规则，哪些还没有规则
    commits_with_existing_rules = []
    commits_to_process = []
    
    for commit in top_n_commits:
        if has_valid_semgrep_rules(commit):
            commits_with_existing_rules.append(commit)
        else:
            commits_to_process.append(commit)
    
    # 统计已有规则数量（只统计前n个中的）
    existing_rules_count = len(commits_with_existing_rules)
    cluster_result["commits_with_existing_rules"] = existing_rules_count
    
    print_progress(f"Cluster {cluster_id} ({cluster_index + 1}/{total_clusters}) - "
                  f"Top {len(top_n_commits)} commits: {existing_rules_count} with existing rules, "
                  f"{len(commits_to_process)} need processing")
    
    if not commits_to_process:
        cluster_result["total_commits_with_rules"] = existing_rules_count
        print_progress(f"Cluster {cluster_id} ({cluster_index + 1}/{total_clusters}) completed - "
                      f"Top {len(top_n_commits)} commits all have rules ({existing_rules_count}/{len(top_n_commits)})")
        return cluster_result
    
    total_commits_to_process = len(commits_to_process)
    print_progress(f"Cluster {cluster_id} ({cluster_index + 1}/{total_clusters}) processing {total_commits_to_process} new commits")
    
    # 线程安全计数器
    processed_counter = ThreadSafeCounter()
    successful_counter = ThreadSafeCounter()
    rules_counter = ThreadSafeCounter()
    
    # 并行处理commits
    with ThreadPoolExecutor(max_workers=generate_config.SIMPLE_COMMIT_MAX_WORKERS) as executor:
        # 提交所有commit任务
        future_to_commit = {}
        for commit_idx, commit in enumerate(commits_to_process, 1):
            future = executor.submit(process_single_commit, commit, cluster_id, commit_idx, total_commits_to_process)
            future_to_commit[future] = commit
        
        # 收集结果
        for future in as_completed(future_to_commit):
            success, commit_result = future.result()
            cluster_result["commit_details"].append(commit_result)
            
            processed_counter.increment()
            if success:
                successful_counter.increment()
            
            rules_generated = commit_result["rules_generated"]
            rules_counter.increment(rules_generated)
            
            # 更新聚类进度（非详细模式下显示）
            if not generate_config.SIMPLE_SHOW_COMMIT_PROGRESS:
                processed = processed_counter.value
                successful = successful_counter.value
                total_rules = rules_counter.value
                print_progress(f"Cluster {cluster_id}: {processed}/{total_commits_to_process} commits processed, "
                              f"{successful} successful, {total_rules} rules generated")
    
    # 更新最终统计
    cluster_result["commits_newly_processed"] = processed_counter.value
    cluster_result["commits_newly_successful"] = successful_counter.value
    cluster_result["total_commits_with_rules"] = existing_rules_count + successful_counter.value
    cluster_result["total_rules_generated"] = rules_counter.value
    
    print_progress(f"Cluster {cluster_id} ({cluster_index + 1}/{total_clusters}) completed - "
                  f"Existing: {existing_rules_count}, New successful: {successful_counter.value}/"
                  f"{processed_counter.value}, Total with rules: {cluster_result['total_commits_with_rules']}, "
                  f"New rules generated: {rules_counter.value}")
    
    return cluster_result

def process_clusters_parallel(clusters, max_workers=None):
    """并行处理多个聚类"""
    if max_workers is None:
        max_workers = generate_config.SIMPLE_CLUSTER_MAX_WORKERS
    
    results = []
    clusters_to_process = clusters
    if generate_config.SIMPLE_PROCESS_CLUSTER_LIMIT > 0:
        clusters_to_process = clusters[:generate_config.SIMPLE_PROCESS_CLUSTER_LIMIT]
    
    if not clusters_to_process:
        return results
    
    total_clusters = len(clusters_to_process)
    print_info(f"Starting parallel processing of {total_clusters} clusters with {max_workers} workers")
    
    # tqdm配置 - 始终显示聚类级进度
    tqdm_kwargs = {
        'desc': "Processing clusters",
        'unit': "cluster",
        'total': total_clusters,
        'dynamic_ncols': True,
        'file': sys.stdout,
        'position': 0,
        'leave': True
    }
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cluster = {}
        for i, cluster in enumerate(clusters_to_process):
            future = executor.submit(process_single_cluster, cluster, i, total_clusters)
            future_to_cluster[future] = i
        
        print_info("All cluster tasks submitted, waiting for completion...")
        
        # 收集结果，显示进度条
        with tqdm(**tqdm_kwargs) as pbar:
            for future in as_completed(future_to_cluster):
                cluster_idx = future_to_cluster[future]
                try:
                    cluster_result = future.result()
                    results.append(cluster_result)
                except Exception as e:
                    # 创建错误结果
                    error_result = {
                        "cluster_id": f"cluster_{cluster_idx}",
                        "cluster_index": cluster_idx,
                        "commits_with_existing_rules": 0,
                        "commits_newly_processed": 0,
                        "commits_newly_successful": 0,
                        "total_commits_with_rules": 0,
                        "total_rules_generated": 0,
                        "commit_details": [],
                        "error_message": f"Cluster processing failed: {str(e)}"
                    }
                    results.append(error_result)
                    print_progress(f"Cluster {cluster_idx} failed with exception: {str(e)}")
                finally:
                    pbar.update(1)
                    pbar.refresh()
    
    # 按cluster_index排序
    results.sort(key=lambda x: x['cluster_index'])
    print_info("All clusters processing completed")
    return results

def generate_overall_statistics(cluster_results):
    """生成整体统计信息"""
    total_clusters = len(cluster_results)
    processed_clusters = sum(1 for r in cluster_results if r['commits_newly_processed'] > 0)
    
    total_commits_with_existing_rules = sum(r['commits_with_existing_rules'] for r in cluster_results)
    total_commits_newly_processed = sum(r['commits_newly_processed'] for r in cluster_results)
    total_commits_newly_successful = sum(r['commits_newly_successful'] for r in cluster_results)
    total_commits_with_rules = sum(r['total_commits_with_rules'] for r in cluster_results)
    total_rules_generated = sum(r['total_rules_generated'] for r in cluster_results)
    
    return {
        "total_clusters": total_clusters,
        "processed_clusters": processed_clusters,
        "total_commits_with_existing_rules": total_commits_with_existing_rules,
        "total_commits_newly_processed": total_commits_newly_processed,
        "total_commits_newly_successful": total_commits_newly_successful,
        "total_commits_with_rules": total_commits_with_rules,
        "total_rules_generated": total_rules_generated,
        "success_rate_new_commits": total_commits_newly_successful / total_commits_newly_processed if total_commits_newly_processed > 0 else 0
    }

def save_processing_result(result_data, output_path):
    """保存处理结果到JSON文件"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=generate_config.DEFAULT_OUTPUT_INDENT, ensure_ascii=False)
        print_info(f"Processing result saved to: {output_path}")
        return True
    except Exception as e:
        print_info(f"Error saving processing result: {e}")
        return False

def print_detailed_statistics(overall_statistics, cluster_results, duration_seconds):
    """打印详细的统计信息（中文）"""
    print_info(f"\n{'='*80}")
    print_info(f"处理结果统计")
    print_info(f"{'='*80}")
    
    # 整体统计
    print_info(f"【整体统计】")
    print_info(f"  聚类总数: {overall_statistics['total_clusters']}")
    print_info(f"  已处理聚类: {overall_statistics['processed_clusters']}")
    print_info(f"  已有规则的 Commit 数: {overall_statistics['total_commits_with_existing_rules']}")
    print_info(f"  新处理的 Commit 数: {overall_statistics['total_commits_newly_processed']}")
    print_info(f"  新处理成功的 Commit 数: {overall_statistics['total_commits_newly_successful']}")
    print_info(f"  新处理成功率: {overall_statistics['success_rate_new_commits']:.1%}")
    print_info(f"  有规则的总 Commit 数: {overall_statistics['total_commits_with_rules']}")
    print_info(f"  新生成的规则总数: {overall_statistics['total_rules_generated']}")
    print_info(f"  总耗时: {duration_seconds} 秒")
    print_info(f"")
    
    # 每个聚类的详细统计
    print_info(f"【各聚类详细结果】")
    for cluster_result in cluster_results:
        cluster_id = cluster_result.get('cluster_id', cluster_result.get('cluster_index', 'Unknown'))
        print_info(f"  聚类 {cluster_id}:")
        print_info(f"    已有规则的 Commit: {cluster_result['commits_with_existing_rules']}")
        print_info(f"    新处理的 Commit: {cluster_result['commits_newly_processed']}")
        print_info(f"    新处理成功的 Commit: {cluster_result['commits_newly_successful']}")
        print_info(f"    有规则的总 Commit 数: {cluster_result['total_commits_with_rules']}")
        print_info(f"    新生成的规则数: {cluster_result['total_rules_generated']}")
        
        # 显示每个 commit 的处理结果
        if cluster_result.get('commit_details'):
            print_info(f"    Commit 详情:")
            for commit_detail in cluster_result['commit_details']:
                repo = commit_detail['repository_name']
                commit_hash = commit_detail['hash'][:8]
                rules_count = commit_detail.get('rules_generated', 0)
                success = commit_detail.get('generation_successful', False)
                status = "✅ 成功" if success else "❌ 失败"
                error = commit_detail.get('error_message')
                print_info(f"      {repo}:{commit_hash} - {status} (生成 {rules_count} 个规则)")
                if error:
                    print_info(f"        错误: {error}")
        print_info(f"")
    
    print_info(f"{'='*80}")

def process_cluster_file_simple(cluster_json_path):
    """处理聚类JSON文件的主函数（简化版）"""
    # 记录开始时间
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    
    print_info(f"处理开始于 {start_time_str}")
    
    # 验证输入文件
    if not os.path.exists(cluster_json_path):
        print_info(f"错误: 未找到聚类 JSON 文件: {cluster_json_path}")
        return None
    
    # 读取聚类数据
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
    
    # 并行处理聚类
    cluster_results = process_clusters_parallel(clusters)
    
    # 记录结束时间和计算耗时
    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    duration_seconds = int(end_time - start_time)
    print_info(f"处理完成于 {end_time_str} (耗时: {duration_seconds} 秒)")
    
    # 生成整体统计
    overall_statistics = generate_overall_statistics(cluster_results)
    
    # 输出详细统计信息
    print_detailed_statistics(overall_statistics, cluster_results, duration_seconds)
    
    return {
        "overall_statistics": overall_statistics,
        "cluster_results": cluster_results
    }

def main():
    """主函数"""
    # 测试文件路径
    CLUSTER_JSON_PATH = global_config.root_path + "python/3-cluster_huawei_stage2/result_final/0_8_2_merged_order.json"
    
    # 输出运行配置
    print_info(f"{'='*80}")
    print_info(f"SEMGREP 规则生成处理 - 配置信息")
    print_info(f"{'='*80}")
    print_verbose(f"聚类 JSON 文件: {CLUSTER_JSON_PATH}")
    print_verbose(f"知识库路径: {os.path.join(global_config.root_path, generate_config.KNOWLEDGE_BASE_RELATIVE_PATH)}")
    print_info(f"处理聚类数量限制: {generate_config.SIMPLE_PROCESS_CLUSTER_LIMIT if generate_config.SIMPLE_PROCESS_CLUSTER_LIMIT > 0 else '全部'}")
    print_info(f"每个聚类处理的 Commit 数: {generate_config.SIMPLE_COMMITS_PER_CLUSTER}")
    print_info(f"聚类分析最大并行数: {generate_config.SIMPLE_CLUSTER_MAX_WORKERS}")
    print_info(f"Commit 分析最大并行数: {generate_config.SIMPLE_COMMIT_MAX_WORKERS}")
    print_info(f"每个 Commit 生成的规则数: {generate_config.COMMIT_GENERATION_COUNT}")
    print_info(f"复用已有规则: {generate_config.COMMIT_REUSE_EXISTING}")
    print_info(f"显示 Commit 进度: {generate_config.SIMPLE_SHOW_COMMIT_PROGRESS}")
    print_info(f"详细输出模式: {generate_config.VERBOSE}")
    print_info(f"{'='*80}")
    print_info("")
    
    result = process_cluster_file_simple(
        cluster_json_path=CLUSTER_JSON_PATH
    )
    
    if result:
        print_info(f"\n处理完成，共处理 {len(result['cluster_results'])} 个聚类")

if __name__ == "__main__":
    main()