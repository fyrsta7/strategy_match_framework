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

# 配置日志记录
logging.basicConfig(
    filename='../../log/evaluate_optimizations.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 每个 commit 上重复运行 performance test 的次数
PERFORMANCE_TEST_NUM = 10

# 全局变量：控制测试数据的重新生成行为
# 如果为 True，忽略现有测试数据，重新运行所有测试并覆盖结果
# 如果为 False，继续基于现有数据运行未完成的测试次数
FORCE_REGENERATION = True  # 根据需要修改此值

# 全局变量：设置当前选择的模型列表
# 可选值：'deepseek-v3', 'deepseek-reasoner', 'gpt-4o', 'o1-mini'
CURRENT_MODELS = ['gpt-4o']  # 根据需要修改此列表

# 保留原本的 supported_models 变量，用于指定所有可以选择的 llm
SUPPORTED_MODELS = ['deepseek-v3', 'deepseek-reasoner', 'gpt-4o', 'o1-mini']

# 检查每个模型是否在支持的模型列表中
for model in CURRENT_MODELS:
    if model not in SUPPORTED_MODELS:
        logging.error(f"不支持的模型 '{model}'。支持的模型有：{SUPPORTED_MODELS}")
        raise ValueError(f"不支持的模型 '{model}'。支持的模型有：{SUPPORTED_MODELS}")

# 存储测试结果的json文件的路径
JSON_DIR_TEMPLATE = os.path.join(config.root_path, "benchmark", "rocksdb", "llm", "baseline")
JSON_FILENAME_TEMPLATE = "baseline_{model}_2.json"

# 存储优化后函数的文件路径
AFTER_FUNC_DIR_NAME = "baseline"
AFTER_FUNC_FILENAME_TEMPLATE = "after_func_{model}_2.txt"

# 确保结果目录存在
os.makedirs(JSON_DIR_TEMPLATE, exist_ok=True)

ROCKSDB_PATH = os.path.join(config.root_path, "repository", "rocksdb")

# 正则模式用于提取基准测试结果
BENCHMARK_PATTERN_NEW = re.compile(
    r'(\w+)\s*:\s*([\d\.]+)\s*micros/op\s*([\d\.]+)\s*ops/sec\s*([\d\.]+)\s*seconds\s*([\d,]+)\s*operations;*\s*([\d\.]+)?\s*MB/s(?:\s*\((\d+) of (\d+) found\))?'
)

BENCHMARK_PATTERN_OLD = re.compile(
    r'(\w+)\s*:\s*([\d\.]+)\s*micros/op\s*([\d\.]+)\s*ops/sec;\s*([\d\.]+)\s*MB/s(?:\s*\((\d+) of (\d+) found\))?'
)

def save_json(data: List[Dict], current_model: str, repo_name: str = "rocksdb"):
    """将数据保存回 JSON 文件"""
    filename = JSON_FILENAME_TEMPLATE.format(model=current_model)
    file_path = os.path.join(JSON_DIR_TEMPLATE, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logging.info(f"成功保存 JSON 数据到 {file_path}。")
    except Exception as e:
        logging.error(f"写入 JSON 文件时出错: {e}")

def load_json(current_model: str, repo_name: str = "rocksdb") -> List[Dict]:
    """从 JSON 文件加载提交数据"""
    filename = JSON_FILENAME_TEMPLATE.format(model=current_model)
    file_path = os.path.join(JSON_DIR_TEMPLATE, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                commits = json.load(f)
                logging.info(f"成功加载 JSON 数据从 {file_path}。")
                return commits
            except json.JSONDecodeError as e:
                logging.error(f"JSON 文件解析失败: {e}")
                return []
    else:
        logging.warning(f"JSON 文件不存在: {file_path}，将初始化为空列表。")
        return []

def parse_benchmark_output(output: str, is_old: bool=False) -> Dict:
    """解析 db_bench 输出"""
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
                if len(match.groups()) < 6:
                    continue
                (bench_name, micros_op, ops_sec, mb_per_sec, found, total_found) = match.groups()[:6]
                benchmarks[bench_name] = {
                    'micros_per_op': float(micros_op),
                    'ops_per_sec': float(ops_sec),
                    'MB_per_sec': float(mb_per_sec)
                }
                if found and total_found:
                    benchmarks[bench_name]['found'] = int(found)
                    benchmarks[bench_name]['total_found'] = int(total_found)
            else:
                if len(match.groups()) < 8:
                    continue
                (bench_name, micros_op, ops_sec, seconds, operations, mb_per_sec, found, total_found) = match.groups()[:8]
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

def clear_test_dir():
    """清空测试目录"""
    TEST_DIR = '/tmp/rocksdb_test'
    try:
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        os.makedirs(TEST_DIR, exist_ok=True)
        logging.info(f"清空并重新创建测试目录: {TEST_DIR}")
    except Exception as e:
        logging.error(f"清空测试目录时出错: {e}")
        raise e

def run_performance_test(rocksdb_path: str, is_old: bool=False) -> Dict:
    """运行性能测试并解析输出"""
    clear_test_dir()
    TEST_DIR = '/tmp/rocksdb_test'
    try:
        result = subprocess.run(['./db_bench', '--benchmarks=fillseq,readrandom',
                                 f'--db={TEST_DIR}', '--num=1000000'],
                                capture_output=True, text=True, cwd=rocksdb_path, check=True)
        logging.info("性能测试运行成功。")
        return parse_benchmark_output(result.stdout, is_old=is_old)
    except subprocess.CalledProcessError as e:
        logging.error(f"运行性能测试时出错: {e}")
        return {}

def get_commit_date(commit_hash: str, rocksdb_path: str) -> Optional[datetime]:
    """获取指定提交的日期"""
    try:
        result = subprocess.check_output(['git', 'show', '-s', '--format=%cI', commit_hash],
                                         cwd=rocksdb_path).decode().strip()
        commit_date = datetime.fromisoformat(result)
        return commit_date
    except subprocess.CalledProcessError as e:
        logging.error(f"获取提交 {commit_hash} 日期时出错: {e}")
        return None
    except ValueError as ve:
        logging.error(f"解析提交 {commit_hash} 日期时出错: {ve}")
        return None

def build_rocksdb(commit_hash: str, use_gcc10: bool, rocksdb_path: str):
    """编译 RocksDB"""
    try:
        # 运行 'make clean'，可能使用 gcc-10
        if use_gcc10:
            subprocess.run(['make', 'clean', 'CC=gcc-10', 'CXX=g++-10'], check=True, cwd=rocksdb_path)
            logging.info(f"运行 'make clean CC=gcc-10 CXX=g++-10' 于提交 {commit_hash}")
        else:
            subprocess.run(['make', 'clean'], check=True, cwd=rocksdb_path)
            logging.info(f"运行 'make clean' 于提交 {commit_hash}")
    except subprocess.CalledProcessError as e:
        logging.error(f"在提交 {commit_hash} 上运行 'make clean' 时出错: {e}")
        raise e

    try:
        # 运行 'make db_bench'
        if use_gcc10:
            subprocess.run(['make', 'db_bench', f'-j{os.cpu_count()}', 'DEBUG_LEVEL=0',
                            'CC=gcc-10', 'CXX=g++-10'], check=True, cwd=rocksdb_path)
            logging.info(f"运行 'make db_bench -j{os.cpu_count()} DEBUG_LEVEL=0 CC=gcc-10 CXX=g++-10' 于提交 {commit_hash}")
        else:
            subprocess.run(['make', 'db_bench', f'-j{os.cpu_count()}', 'DEBUG_LEVEL=0'], check=True, cwd=rocksdb_path)
            logging.info(f"运行 'make db_bench -j{os.cpu_count()} DEBUG_LEVEL=0' 于提交 {commit_hash}")
    except subprocess.CalledProcessError as e:
        logging.error(f"在提交 {commit_hash} 上运行 'make db_bench' 时出错: {e}")
        raise e

def apply_optimized_function(commit: Dict, repo_name: str, current_model: str, rocksdb_path: str) -> bool:
    """
    将 LLM 优化后的函数替换到父提交的代码库中。
    返回替换是否成功的布尔值。
    """
    commit_hash = commit.get("hash")
    if not commit_hash:
        logging.warning("提交记录缺少 'hash' 字段，跳过。")
        return False

    modified_files = commit.get("modified_files", [])
    modified_functions = commit.get("modified_functions", [])

    if not modified_files or not modified_functions:
        logging.warning(f"提交 {commit_hash} 缺少 'modified_files' 或 'modified_functions' 字段，跳过。")
        return False

    # 检查 before_func.txt 和 after_func.txt 是否存在
    before_func_path = os.path.join(config.root_path, "result", repo_name, "modified_file", commit_hash, "before_func.txt")
    optimized_func_path = os.path.join(config.root_path, "benchmark", repo_name, "modified_file", commit_hash, AFTER_FUNC_DIR_NAME, AFTER_FUNC_FILENAME_TEMPLATE.format(model = current_model))

    if not os.path.exists(optimized_func_path) or not os.path.exists(before_func_path):
        logging.warning(f"提交 {commit_hash} 缺少 'before_func.txt' 或 'after_func.txt'，跳过测试。")
        return False

    # 假设每个提交只有一个修改的文件和一个修改的函数
    # 如果有多个，需进一步处理
    if len(modified_files) != 1 or len(modified_functions) != 1:
        logging.warning(f"提交 {commit_hash} 有多个修改的文件或函数，当前脚本仅支持单个修改，跳过。")
        return False

    modified_file = modified_files[0]
    modified_function = modified_functions[0]
    function_name = modified_function.get("name")
    start_line = modified_function.get("start_line")
    end_line = modified_function.get("end_line")

    if not function_name or not start_line or not end_line:
        logging.warning(f"提交 {commit_hash} 的修改函数缺少 'name'、'start_line' 或 'end_line' 字段，跳过。")
        return False

    # 定位源代码文件
    source_file_path = os.path.join(rocksdb_path, modified_file)
    if not os.path.exists(source_file_path):
        logging.error(f"源代码文件不存在: {source_file_path}，提交 {commit_hash}，跳过。")
        return False

    # 读取优化后的函数内容
    try:
        with open(optimized_func_path, 'r', encoding='utf-8') as f:
            optimized_function_code = f.read()
    except Exception as e:
        logging.error(f"读取优化后的函数文件 {optimized_func_path} 时出错: {e}，提交 {commit_hash}，跳过。")
        return False

    # 读取源代码文件内容
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"源代码文件不存在: {source_file_path}，提交 {commit_hash}，跳过。")
        return False

    # 替换函数内容
    try:
        # 调整索引为0基
        new_source_lines = source_lines[:start_line - 1] + [optimized_function_code + '\n'] + source_lines[end_line:]
        with open(source_file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_source_lines)
        logging.info(f"已将优化后的函数 '{function_name}' 替换到文件 {modified_file} 中，提交 {commit_hash}，模型 {current_model}")
        return True
    except Exception as e:
        logging.error(f"替换函数内容时出错: {e}，提交 {commit_hash}，模型 {current_model}，跳过。")
        return False

def ensure_test_runs(commits: List[Dict], commit: Dict, key: str, current_model: str, is_old: bool=False):
    """
    确保每个提交有足够的测试运行次数。
    记录到新的 LLM 优化结果字段中。
    """
    if FORCE_REGENERATION:
        # 如果强制重新生成，重置测试数据字段
        commit[key] = []
        commit[f'llm_average_child_{current_model}'] = {}
        commit[f'llm_comparison_child_to_parent_{current_model}'] = {}
        logging.info(f"强制重新生成测试数据，已清空提交 {commit.get('hash')} 的现有测试结果。")

    existing_runs = commit.get(key, [])
    runs_needed = PERFORMANCE_TEST_NUM - len(existing_runs)

    if runs_needed <= 0 and not FORCE_REGENERATION:
        logging.info(f"{key} 已经有 {PERFORMANCE_TEST_NUM} 组或更多数据，跳过运行测试。")
        return

    for run_number in range(len(existing_runs) + 1, PERFORMANCE_TEST_NUM + 1):
        logging.info(f"运行性能测试 {run_number} 于提交 {commit.get('hash')} 的 {key[:-7]}，模型 {current_model}")
        benchmark_result = run_performance_test(rocksdb_path=ROCKSDB_PATH, is_old=is_old)
        commit.setdefault(key, []).append(benchmark_result)
        time.sleep(1)  # 可选：等待以确保系统稳定性

        # 每次运行后保存 JSON，以防数据丢失
        save_json(commits, current_model=current_model, repo_name="rocksdb")
        logging.info(f"已保存第 {run_number} 次测试结果于提交 {commit.get('hash')}，模型 {current_model}")

def calculate_average(test_runs: List[Dict]) -> Dict:
    """计算测试运行的平均值"""
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

def compare_averages(child_avg: Dict, parent_avg: Dict) -> Dict:
    """比较子提交和父提交的平均值"""
    comparison = {}
    benchmarks = child_avg.keys() & parent_avg.keys()
    for bench in benchmarks:
        comparison[bench] = {}
        metrics = child_avg[bench].keys() & parent_avg[bench].keys()
        for metric in metrics:
            child_value = child_avg[bench][metric]
            parent_value = parent_avg[bench][metric]
            if isinstance(child_value, (int, float)) and isinstance(parent_value, (int, float)):
                if child_value > parent_value:
                    comparison[bench][metric] = 'increased'
                elif child_value < parent_value:
                    comparison[bench][metric] = 'decreased'
                else:
                    comparison[bench][metric] = 'no change'
            else:
                comparison[bench][metric] = 'no change'
    return comparison

def process_commit(commit: Dict, repo_name: str, commits: List[Dict], current_model: str, rocksdb_path: str) -> Dict:
    """处理单个提交，包括应用优化、编译、测试和记录结果"""
    commit_hash = commit.get('hash')
    if not commit_hash:
        logging.warning("提交记录缺少 'hash' 字段，跳过。")
        return commit

    # 使用新增字段来避免与已有字段冲突
    llm_performance_test_child = f'llm_performance_test_child'
    llm_average_child = f'llm_average_child'
    llm_comparison_child_to_parent = f'llm_comparison_child_to_parent'

    # 获取父提交
    try:
        parent_hash = subprocess.check_output(['git', 'rev-parse', f'{commit_hash}~1'], cwd=rocksdb_path).decode().strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"获取父提交 {commit_hash} 时出错: {e}")
        return commit

    # 回退到父提交
    try:
        subprocess.run(['git', 'reset', '--hard'], check=True, cwd=rocksdb_path)
        subprocess.run(['git', 'checkout', parent_hash], check=True, cwd=rocksdb_path)
        logging.info(f"已切换到父提交 {parent_hash} 且重置文件。")
    except subprocess.CalledProcessError as e:
        logging.error(f"切换到父提交 {parent_hash} 时出错: {e}")
        return commit

    # 应用优化后的函数
    success = apply_optimized_function(commit, repo_name, current_model, rocksdb_path)
    if not success:
        logging.error(f"应用优化函数失败于提交 {commit_hash}，模型 {current_model}，跳过后续步骤。")
        return commit

    # 编译 RocksDB
    commit_date = get_commit_date(parent_hash, rocksdb_path)
    if commit_date is None:
        logging.error(f"获取父提交 {parent_hash} 日期失败，跳过编译和测试。")
        return commit

    use_gcc10 = commit_date < datetime.fromisoformat("2023-06-02T16:39:14-07:00")
    try:
        build_rocksdb(parent_hash, use_gcc10, rocksdb_path)
    except Exception as e:
        logging.error(f"编译 RocksDB 于父提交 {parent_hash} 时失败: {e}")
        return commit

    # 运行性能测试（LLM 优化后的父提交）
    is_old = commit_date <= datetime.fromisoformat("2022-03-21T17:30:51-07:00")
    ensure_test_runs(commits, commit, llm_performance_test_child, current_model=current_model, is_old=is_old)

    # 计算平均值
    commit[llm_average_child] = calculate_average(commit.get(llm_performance_test_child, []))

    # 获取父提交的测试结果
    parent_commit = next((c for c in commits if c.get('hash') == parent_hash), None)
    parent_average_key = f'llm_average_child_{current_model}'
    if parent_commit and parent_commit.get(parent_average_key):
        # 比较测试结果
        commit[llm_comparison_child_to_parent] = compare_averages(
            commit.get(llm_average_child, {}),
            parent_commit.get(parent_average_key, {})
        )
    else:
        logging.warning(f"父提交 {parent_hash} 没有可比较的测试结果。")
        commit[llm_comparison_child_to_parent] = {}

    # 保存结果
    save_json(commits, current_model=current_model, repo_name=repo_name)

    return commit

def main():
    rocksdb_path = os.path.join(config.root_path, "repository", "rocksdb")
    make_path = shutil.which('make')
    if make_path is None:
        logging.error("Make not found in PATH")
        raise Exception("Make not found in PATH")

    # 遍历每个模型
    for current_model in CURRENT_MODELS:
        logging.info(f"开始处理模型：{current_model}")

        # 定义路径
        JSON_PATH = os.path.join(JSON_DIR_TEMPLATE, JSON_FILENAME_TEMPLATE.format(model=current_model))

        # 加载提交数据
        commits = load_json(current_model=current_model, repo_name="rocksdb")

        # 遍历每个提交
        for commit in commits:
            commit_hash = commit.get('hash')
            if not commit_hash:
                logging.warning("找到一个没有 'hash' 字段的提交，跳过。")
                continue
                
            logging.info(f"开始处理提交 {commit_hash}，模型 {current_model}")

            # 检查是否已经有LLM优化的测试数据
            if not FORCE_REGENERATION:
                existing_runs = commit.get("llm_performance_test_child", [])
                logging.info(f"existing_runs: {len(existing_runs)}")
                if len(existing_runs) >= PERFORMANCE_TEST_NUM:
                    logging.info(f"提交 {commit_hash} 已经有足够的LLM优化测试数据，模型 {current_model}，跳过。")
                    continue  # 跳过该提交，避免后续操作

            # 检查是否存在 before_func.txt 和 after_func.txt
            before_func_path = os.path.join(config.root_path, "result", "rocksdb", "modified_file", commit_hash, "before_func.txt")
            after_func_path = os.path.join(config.root_path, "benchmark", "rocksdb", "modified_file", commit_hash, AFTER_FUNC_DIR_NAME, AFTER_FUNC_FILENAME_TEMPLATE.format(model = current_model))

            logging.debug(f"检查文件: {before_func_path} 和 {after_func_path}")

            if not os.path.exists(before_func_path):
                logging.warning(f"提交 {commit_hash} 缺少 'before_func.txt'，跳过测试。")
            if not os.path.exists(after_func_path):
                logging.warning(f"提交 {commit_hash} 缺少 'after_func.txt'，跳过测试。")

            if not os.path.exists(before_func_path) or not os.path.exists(after_func_path):
                continue

            # 处理提交
            process_commit(commit, repo_name="rocksdb", commits=commits, current_model=current_model, rocksdb_path=rocksdb_path)

        logging.info(f"模型 {current_model} 的所有提交处理已完成。")

    logging.info("所有模型的处理已完成。")

if __name__ == "__main__":
    main()