import json
import subprocess
import os
import shutil
import logging
import re
import time
from statistics import mean
from typing import List, Optional, Dict
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Configure logging
logging.basicConfig(filename='../../log/script.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 每个 commit 上重复运行 performance test 的次数
performance_test_num = 1

# 是否忽略已有测试结果重新测试
# 如果为True，则忽略现有结果，重新测试每个commit
# 如果为False，则在现有结果基础上继续测试(如果已完成则跳过)
ignore_existing_results = False

# Define paths
JSON_PATH = config.root_path + "benchmark/rocksdb/test_result.json"
FILTERED_JSON_PATH = config.root_path + "benchmark/rocksdb/filtered_test_result.json"
ROCKSDB_PATH = config.root_path + "repository/rocksdb"
TEST_DIR = '/tmp/rocksdb_test'

MAKE_PATH = shutil.which('make')
if MAKE_PATH is None:
    raise Exception("Make not found in PATH")

# Function to save commits data to JSON
def save_json():
    try:
        with open(JSON_PATH, 'w') as f:
            json.dump(commits, f, indent=4)
        logging.info("Successfully saved JSON data.")
    except Exception as e:
        logging.error(f"Error writing to JSON file: {e}")

# Function to save filtered commits data to JSON
def save_filtered_json(filtered_commits):
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(FILTERED_JSON_PATH), exist_ok=True)
        
        # 如果文件已存在，先读取现有内容
        existing_commits = []
        if os.path.exists(FILTERED_JSON_PATH):
            with open(FILTERED_JSON_PATH, 'r') as f:
                try:
                    existing_commits = json.load(f)
                except json.JSONDecodeError:
                    logging.error("Filtered JSON 文件解析失败。将覆盖。")
        
        # 合并现有commits和新的filtered_commits，避免重复
        existing_hashes = {commit.get('hash') for commit in existing_commits if 'hash' in commit}
        for commit in filtered_commits:
            if commit.get('hash') and commit.get('hash') not in existing_hashes:
                existing_commits.append(commit)
                existing_hashes.add(commit.get('hash'))
        
        # 保存合并后的结果
        with open(FILTERED_JSON_PATH, 'w') as f:
            json.dump(existing_commits, f, indent=4)
        logging.info(f"Successfully saved filtered JSON data with {len(existing_commits)} commits.")
    except Exception as e:
        logging.error(f"Error writing to filtered JSON file: {e}")

# Read JSON file
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'r') as f:
        try:
            commits = json.load(f)
            logging.info("Successfully loaded JSON data.")
        except json.JSONDecodeError:
            logging.error("JSON 文件解析失败。请检查文件格式。")
            commits = []
else:
    logging.warning(f"JSON 文件不存在：{JSON_PATH}，将初始化为空列表。")
    commits = []

# Enter RocksDB repository directory
os.chdir(ROCKSDB_PATH)

# Define the cutoff datetimes
CUTOFF_DATETIME_STR = "2023-06-02T16:39:14-07:00"
CUTOFF_DATETIME = datetime.fromisoformat(CUTOFF_DATETIME_STR)
EXTREME_CUTOFF_DATETIME_STR = "2022-03-21T17:30:51-07:00"
EXTREME_CUTOFF_DATETIME = datetime.fromisoformat(EXTREME_CUTOFF_DATETIME_STR)

# Regex patterns to extract benchmark results
BENCHMARK_PATTERN_NEW = re.compile(
    r'(\w+)\s*:\s*([\d\.]+)\s*micros/op\s*([\d\.]+)\s*ops/sec\s*([\d\.]+)\s*seconds\s*([\d,]+)\s*operations;*\s*([\d\.]+)?\s*MB/s(?:\s*\((\d+) of (\d+) found\))?'
)
BENCHMARK_PATTERN_OLD = re.compile(
    r'(\w+)\s*:\s*([\d\.]+)\s*micros/op\s*([\d\.]+)\s*ops/sec;\s*([\d\.]+)\s*MB/s(?:\s*\((\d+) of (\d+) found\))?'
)

# Function to clear the test directory
def clear_test_dir():
    subprocess.run(['rm', '-rf', f'{TEST_DIR}/*'], check=True)
    logging.info(f"Cleared test directory {TEST_DIR}")

# Function to run performance test and parse output
def run_performance_test(is_old=False):
    clear_test_dir()
    try:
        result = subprocess.run(['./db_bench', '--benchmarks=fillseq,readrandom',
                                 f'--db={TEST_DIR}', '--num=1000000'],
                                capture_output=True, text=True, check=True)
        logging.info("Performance test run successfully.")
        return parse_benchmark_output(result.stdout, is_old)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running performance test: {e}")
        return {}

# Function to parse db_bench output
def parse_benchmark_output(output, is_old=False):
    benchmarks = {}
    lines = output.splitlines()
    for line in lines:
        line = line.strip()
        if is_old:
            match = BENCHMARK_PATTERN_OLD.match(line)
        else:
            match = BENCHMARK_PATTERN_NEW.match(line)
        if match:
            if is_old:
                (bench_name, micros_op, ops_sec, mb_per_sec, found, total_found) = match.groups()
                benchmarks[bench_name] = {
                    'micros_per_op': float(micros_op),
                    'ops_per_sec': float(ops_sec),
                    'MB_per_sec': float(mb_per_sec)
                }
                if found and total_found:
                    benchmarks[bench_name]['found'] = int(found)
                    benchmarks[bench_name]['total_found'] = int(total_found)
            else:
                (bench_name, micros_op, ops_sec, seconds, operations, mb_per_sec, found, total_found) = match.groups()
                benchmarks[bench_name] = {
                    'micros_per_op': float(micros_op),
                    'ops_per_sec': float(ops_sec),
                    'seconds': float(seconds),
                    'operations': int(operations.replace(',', ''))
                }
                if mb_per_sec:
                    benchmarks[bench_name]['MB_per_sec'] = float(mb_per_sec)
                if found and total_found:
                    benchmarks[bench_name]['found'] = int(found)
                    benchmarks[bench_name]['total_found'] = int(total_found)
    return benchmarks

# Function to get commit date
def get_commit_date(commit_hash: str) -> Optional[datetime]:
    """
    获取指定提交的日期，并返回为 datetime 对象。
    """
    try:
        result = subprocess.check_output(['git', 'show', '-s', '--format=%cI', commit_hash],
                                         cwd=ROCKSDB_PATH).decode().strip()
        commit_date = datetime.fromisoformat(result)
        return commit_date
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting date for commit {commit_hash}: {e}")
        return None
    except ValueError as ve:
        logging.error(f"Error parsing date for commit {commit_hash}: {ve}")
        return None

# Function to build RocksDB
def build_rocksdb(commit_hash: str, use_gcc10: bool):
    try:
        # Run 'make clean' with or without CC and CXX
        if use_gcc10:
            subprocess.run(['make', 'clean', 'CC=gcc-10', 'CXX=g++-10'], check=True)
            logging.info(f"Ran 'make clean CC=gcc-10 CXX=g++-10' for commit {commit_hash}")
        else:
            subprocess.run(['make', 'clean'], check=True)
            logging.info(f"Ran 'make clean' for commit {commit_hash}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running make clean for commit {commit_hash}: {e}")
        raise e
    try:
        # Run 'make db_bench' with or without CC and CXX
        if use_gcc10:
            subprocess.run(['make', 'db_bench', f'-j{os.cpu_count()}', 'DEBUG_LEVEL=0',
                            'CC=gcc-10', 'CXX=g++-10'], check=True)
            logging.info(f"Ran 'make db_bench -j{os.cpu_count()} DEBUG_LEVEL=0 CC=gcc-10 CXX=g++-10' for commit {commit_hash}")
        else:
            subprocess.run(['make', 'db_bench', f'-j{os.cpu_count()}', 'DEBUG_LEVEL=0'], check=True)
            logging.info(f"Ran 'make db_bench -j{os.cpu_count()} DEBUG_LEVEL=0' for commit {commit_hash}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error building RocksDB for commit {commit_hash}: {e}")
        raise e

# Function to ensure there are performance_test_num test runs, running additional tests if necessary
def ensure_test_runs(commit, key, commit_hash, use_gcc10, is_old=False):
    existing_runs = commit.get(key, [])
    
    # 如果配置为忽略现有结果，则重置现有测试结果
    if ignore_existing_results:
        existing_runs = []
        commit[key] = []
        logging.info(f"忽略现有测试结果，重新测试 {commit_hash} 的 {key}")
    
    runs_needed = performance_test_num - len(existing_runs)
    if runs_needed <= 0:
        logging.info(f"{key} 已经有{performance_test_num}组或更多数据，跳过运行测试。")
        return
    
    for run_number in range(len(existing_runs) + 1, performance_test_num + 1):
        logging.info(f"Running performance test {run_number} for {key[:-7]} commit {commit_hash}")
        benchmark_result = run_performance_test(is_old)
        commit.setdefault(key, []).append(benchmark_result)
        time.sleep(1)  # 可选：等待以确保系统稳定性
        # Save after each run to prevent data loss
        save_json()
        logging.info(f"Updated JSON file after run {run_number} for {key[:-7]} commit {commit_hash}")

# Function to calculate average metrics from test runs
def calculate_average(test_runs):
    averages = {}
    if not test_runs:
        return averages
    benchmarks = test_runs[0].keys()
    for bench in benchmarks:
        averages[bench] = {}
        metrics = test_runs[0][bench].keys()
        for metric in metrics:
            metric_values = [run[bench].get(metric) for run in test_runs if run.get(bench) and run[bench].get(metric) is not None]
            if metric_values:
                averages[bench][metric] = mean(metric_values)
    return averages

# Function to compare averages between commit and parent
def compare_averages(child_avg, parent_avg):
    comparison = {}
    benchmarks = child_avg.keys() & parent_avg.keys()
    for bench in benchmarks:
        comparison[bench] = {}
        metrics = child_avg[bench].keys() & parent_avg[bench].keys()
        for metric in metrics:
            child_value = child_avg[bench][metric]
            parent_value = parent_avg[bench][metric]
            if child_value > parent_value:
                comparison[bench][metric] = 'increased'
            elif child_value < parent_value:
                comparison[bench][metric] = 'decreased'
            else:
                comparison[bench][metric] = 'no change'
    return comparison

# 存储符合条件的commits
filtered_commits = []

# Iterate over each commit
for commit in commits:
    commit_hash = commit.get('hash')
    if not commit_hash:
        logging.warning("找到一个没有 'hash' 字段的提交，跳过。")
        continue
    logging.info(f"Processing commit {commit_hash}")
    
    # 检查是否已经有performance_test_num组测试数据
    # 如果有且ignore_existing_results为False，则跳过该提交的测试阶段
    existing_runs_child = commit.get('performance_test_child', [])
    if len(existing_runs_child) >= performance_test_num and not ignore_existing_results:
        logging.info(f"Commit {commit_hash} 已经有{performance_test_num}组测试数据，跳过测试阶段。")
        # 检查是否符合过滤条件：child 和 parent 测试结果都不为空
        if (commit.get('performance_test_child') and commit.get('performance_test_parent') and 
            len(commit.get('performance_test_child', [])) > 0 and 
            len(commit.get('performance_test_parent', [])) > 0):
            logging.info(f"Commit {commit_hash} 符合过滤条件：parent 和 child 测试结果都不为空")
            filtered_commits.append(commit.copy())
        continue  # 跳过当前提交，处理下一个提交

    # Get commit date
    commit_date = get_commit_date(commit_hash)
    if commit_date is None:
        logging.error(f"Skipping commit {commit_hash} due to date retrieval failure.")
        continue

    # Determine whether to use gcc-10 based on commit date
    use_gcc10 = commit_date < CUTOFF_DATETIME
    logging.info(f"Commit {commit_hash} date: {commit_date.isoformat()} - Use gcc-10: {use_gcc10}")

    # Determine if the commit is in the old format
    is_old = commit_date <= EXTREME_CUTOFF_DATETIME
    if is_old:
        logging.info(f"Commit {commit_hash} is old (<= {EXTREME_CUTOFF_DATETIME.isoformat()}), using old parsing logic.")
    else:
        logging.info(f"Commit {commit_hash} is new (> {EXTREME_CUTOFF_DATETIME.isoformat()}), using new parsing logic.")

    # Checkout the commit with reset and clean
    try:
        subprocess.run(['git', 'reset', '--hard'], check=True)
        subprocess.run(['git', 'checkout', commit_hash], check=True)
        logging.info(f"Checked out commit {commit_hash}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking out commit {commit_hash}: {e}")
        continue

    # Build RocksDB for the commit
    try:
        build_rocksdb(commit_hash, use_gcc10)
    except Exception as e:
        logging.error(f"Skipping performance tests for commit {commit_hash} due to build error.")
        commit['performance_test_child'] = commit.get('performance_test_child', [])
        commit['average_child'] = {}
        commit['comparison_child_to_parent'] = {}
        # Save current state to JSON
        save_json()
        continue

    # Initialize performance_test_child if not present
    if 'performance_test_child' not in commit:
        commit['performance_test_child'] = []

    # Ensure there are performance_test_num test runs for the commit
    ensure_test_runs(commit, 'performance_test_child', commit_hash, use_gcc10, is_old=is_old)

    # Calculate average for the commit
    commit['average_child'] = calculate_average(commit['performance_test_child'])

    # Save after processing commit's child tests
    save_json()

    # Get parent commit hash
    try:
        parent_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD~1']).decode().strip()
        logging.info(f"Processing parent commit {parent_hash}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting parent commit of {commit_hash}: {e}")
        parent_hash = None

    if parent_hash:
        # Checkout the parent commit with reset and clean
        try:
            subprocess.run(['git', 'reset', '--hard'], check=True)
            subprocess.run(['git', 'checkout', parent_hash], check=True)
            logging.info(f"Checked out parent commit {parent_hash}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error checking out parent commit {parent_hash}: {e}")
            parent_hash = None

    if parent_hash:
        # Build RocksDB for the parent commit
        try:
            parent_commit_date = get_commit_date(parent_hash)
            if parent_commit_date is None:
                logging.error(f"Skipping parent commit {parent_hash} due to date retrieval failure.")
                raise ValueError(f"Invalid parent commit date for {parent_hash}")

            # Determine whether to use gcc-10 based on parent commit date
            parent_use_gcc10 = parent_commit_date < CUTOFF_DATETIME
            logging.info(f"Parent commit {parent_hash} date: {parent_commit_date.isoformat()} - Use gcc-10: {parent_use_gcc10}")

            # Determine if the parent commit is in the old format
            parent_is_old = parent_commit_date <= EXTREME_CUTOFF_DATETIME
            if parent_is_old:
                logging.info(f"Parent commit {parent_hash} is old (<= {EXTREME_CUTOFF_DATETIME.isoformat()}), using old parsing logic.")
            else:
                logging.info(f"Parent commit {parent_hash} is new (> {EXTREME_CUTOFF_DATETIME.isoformat()}), using new parsing logic.")

            build_rocksdb(parent_hash, parent_use_gcc10)
        except Exception as e:
            logging.error(f"Skipping performance tests for parent commit {parent_hash} due to build error.")
            commit['performance_test_parent'] = commit.get('performance_test_parent', [])
            commit['average_parent'] = {}
            commit['comparison_child_to_parent'] = {}
            # Save current state to JSON
            save_json()
            parent_hash = None

        if parent_hash:
            # Initialize performance_test_parent if not present
            if 'performance_test_parent' not in commit:
                commit['performance_test_parent'] = []

            # Ensure there are performance_test_num test runs for the parent commit
            ensure_test_runs(commit, 'performance_test_parent', parent_hash, parent_use_gcc10, is_old=parent_is_old)

            # Calculate average for the parent commit
            commit['average_parent'] = calculate_average(commit['performance_test_parent'])

            # Compare averages between commit and parent
            commit['comparison_child_to_parent'] = compare_averages(commit['average_child'], commit['average_parent'])

            # 检查是否符合过滤条件：child 和 parent 测试结果都不为空
            if (commit.get('performance_test_child') and commit.get('performance_test_parent') and 
                len(commit.get('performance_test_child', [])) > 0 and 
                len(commit.get('performance_test_parent', [])) > 0):
                logging.info(f"Commit {commit_hash} 符合过滤条件：parent 和 child 测试结果都不为空")
                filtered_commits.append(commit.copy())

            # Save after processing parent commit's tests
            save_json()
        else:
            commit['performance_test_parent'] = commit.get('performance_test_parent', [])
            commit['average_parent'] = {}
            commit['comparison_child_to_parent'] = {}
    else:
        commit['performance_test_parent'] = commit.get('performance_test_parent', [])
        commit['average_parent'] = {}
        commit['comparison_child_to_parent'] = {}

    # Switch back to the commit for further processing
    try:
        subprocess.run(['git', 'reset', '--hard'], check=True)
        subprocess.run(['git', 'checkout', commit_hash], check=True)
        logging.info(f"Checked out commit {commit_hash} again after processing parent")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking out commit {commit_hash} again: {e}")

    # Save after fully processing the commit
    save_json()

# 保存过滤后的commits
if filtered_commits:
    save_filtered_json(filtered_commits)
    logging.info(f"Saved {len(filtered_commits)} filtered commits to {FILTERED_JSON_PATH}")

# Switch back to the original branch (如果需要)
# try:
#     original_branch = subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD'], cwd=ROCKSDB_PATH).decode().strip()
#     subprocess.run(['git', 'checkout', original_branch], check=True)
#     subprocess.run(['git', 'reset', '--hard'], check=True)
#     logging.info(f"Switched back to original branch {original_branch}")
# except subprocess.CalledProcessError as e:
#     logging.error(f"Error switching back to original branch: {e}")

logging.info("All tests completed and results saved.")