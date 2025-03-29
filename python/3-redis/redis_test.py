import os
import re
import subprocess
import time
import signal
import sys
import json
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Configure logging
logging.basicConfig(filename='../../log/redis_test.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 目标测试重复次数
TARGET_TEST_COUNT = 3
# 是否强制重新测试，True 为忽略现有结果重新测试，False 为使用现有结果继续测试
FORCE_FULL_RETEST = False

# ---------------------------
def parse_benchmark_output(output):
    """
    解析 redis-benchmark 输出，提取每个测试块的关键信息：
      · 测试名称（块标题，例如 "====== TEST_NAME ======"）
      · throughput summary（单位为 requests per second）
      · latency summary（avg、min、p50、p95、p99、max）
    返回一个字典，键是测试名称，值为相关信息的字典。
    """
    results = {}
    current_test = None
    lines = output.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 检测测试块标题
        if line.startswith("======") and line.endswith("======"):
            current_test = line.strip("=").strip()
            results[current_test] = {}
        elif current_test is not None:
            # 提取 throughput summary
            if "throughput summary:" in line:
                m = re.search(r"throughput summary:\s*([\d\.]+)\s*requests per second", line)
                if m:
                    results[current_test]['throughput'] = float(m.group(1))
            # 提取 latency summary 信息，下一行为表头、紧接下行为数值
            elif "latency summary (msec):" in line:
                if i + 2 < len(lines):
                    header_line = lines[i+1].strip()  # 表头：avg min p50 p95 p99 max
                    value_line = lines[i+2].strip()
                    parts = value_line.split()
                    if len(parts) >= 6:
                        try:
                            latency_vals = {
                                'avg': float(parts[0]),
                                'min': float(parts[1]),
                                'p50': float(parts[2]),
                                'p95': float(parts[3]),
                                'p99': float(parts[4]),
                                'max': float(parts[5])
                            }
                            results[current_test]['latency'] = latency_vals
                        except Exception as ex:
                            logging.error("解析 latency summary 失败: %s", ex)
                    i += 2  # 跳过已处理的两行
        i += 1
    return results

# ---------------------------
def compile_code(repo_path):
    """
    在 repo_path 目录下编译 Redis 代码（仅编译一次）。
    返回 True 表示编译成功，否则 False。
    """
    os.chdir(repo_path)
    build_cmd = f"make -j{os.cpu_count()} BUILD_TLS=yes"
    logging.info("编译命令：%s", build_cmd)
    print("编译命令：", build_cmd)
    ret = subprocess.run(build_cmd, shell=True)
    if ret.returncode != 0:
        logging.error("编译失败")
        print("编译失败")
        return False
    logging.info("编译成功")
    print("编译成功")
    return True

# ---------------------------
def run_benchmark():
    """
    在 src 目录下启动 redis-server，运行 redis-benchmark 并返回解析后的结果。
    此函数不包括编译过程，应在编译完成后重复调用。
    """
    redis_server_cmd = "./redis-server"
    logging.info("启动 Redis 服务，命令：%s", redis_server_cmd)
    print("启动 Redis 服务器...")
    redis_server_proc = subprocess.Popen(redis_server_cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(2)  # 等待服务器启动
    try:
        redis_benchmark_cmd = "./redis-benchmark"
        logging.info("执行 benchmark，命令：%s", redis_benchmark_cmd)
        bench_proc = subprocess.run(redis_benchmark_cmd, shell=True, capture_output=True, text=True)
        output = bench_proc.stdout
        result = parse_benchmark_output(output)
    finally:
        os.killpg(os.getpgid(redis_server_proc.pid), signal.SIGTERM)
        time.sleep(1)
    return result

# ---------------------------
def run_tests_for_state(repo_path, src_path, existing_results):
    """
    针对当前 git 状态（已切换到对应 commit），先编译一次代码，
    然后根据已有的测试结果数量决定需要额外跑几次测试。
    如果 FORCE_FULL_RETEST 为 True，则忽略已存在的结果从零开始跑。
    返回一个列表，包含 TARGET_TEST_COUNT 次测试结果。
    """
    results = []
    if not FORCE_FULL_RETEST and existing_results:
        results = existing_results
    already = len(results)
    need = TARGET_TEST_COUNT - already
    if need <= 0:
        logging.info("已有 %d 次测试结果，满足要求，跳过测试。", already)
        print(f"已有 {already} 次测试结果，满足要求，跳过测试。")
        return results
    logging.info("需要额外运行 %d 次测试。", need)
    print(f"需要额外运行 {need} 次测试。")
    # 先编译一次
    if not compile_code(repo_path):
        return results
    os.chdir(src_path)
    for i in range(need):
        logging.info("运行第 %d 次测试……", already + i + 1)
        print(f"测试第 {already+i+1} 次……")
        r = run_benchmark()
        if r is not None:
            results.append(r)
    return results

# ---------------------------
def main():
    repo_path = "/raid/zyw/llm_on_code/redis"
    src_path = os.path.join(repo_path, "src")
    benchmark_dir = os.path.join(config.root_path, "benchmark", "redis")
    test_result_file = os.path.join(benchmark_dir, "test_result.json")
    one_func_file = os.path.join(benchmark_dir, "one_func.json")
    
    # 若 test_result.json 不存在，则从 one_func.json 复制并删除 "all_functions" 字段
    if not os.path.exists(test_result_file):
        logging.info("文件 %s 不存在，开始从 %s 复制数据", test_result_file, one_func_file)
        if not os.path.exists(one_func_file):
            logging.error("找不到文件: %s", one_func_file)
            print("错误：找不到", one_func_file)
            sys.exit(1)
        with open(one_func_file, "r", encoding="utf-8") as f:
            commits = json.load(f)
        for commit in commits:
            if "all_functions" in commit:
                del commit["all_functions"]
        os.makedirs(benchmark_dir, exist_ok=True)
        with open(test_result_file, "w", encoding="utf-8") as f:
            json.dump(commits, f, indent=4)
        logging.info("数据已复制到 %s", test_result_file)
    else:
        with open(test_result_file, "r", encoding="utf-8") as f:
            commits = json.load(f)
    
    if not os.path.exists(repo_path):
        logging.error("目录 %s 不存在", repo_path)
        print(f"错误：目录 {repo_path} 不存在。")
        sys.exit(1)
    os.chdir(repo_path)
    logging.info("Repository: %s", repo_path)
    print("Repository:", repo_path)

    # 对每个 commit 依次处理
    for idx, commit in enumerate(commits):
        commit_hash = commit.get("hash")
        if not commit_hash:
            continue

        logging.info("开始处理 commit: %s", commit_hash)
        print("\n处理 commit:", commit_hash)

        # ---------------------------
        # Child 状态测试
        child_existing = commit.get("performance_test_child", [])
        subprocess.run("git reset --hard", shell=True)
        ret = subprocess.run(f"git checkout {commit_hash}", shell=True)
        if ret.returncode != 0:
            logging.error("切换到 commit %s 失败，跳过此 commit", commit_hash)
            print(f"切换到 commit {commit_hash} 失败，跳过。")
            continue
        time.sleep(1)
        logging.info("【Child 状态】开始测试")
        print("【Child 状态】开始测试……")
        child_results = run_tests_for_state(repo_path, src_path, child_existing)
        commit["performance_test_child"] = child_results

        # ---------------------------
        # Parent 状态测试
        proc = subprocess.run(f"git rev-parse {commit_hash}^", shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            logging.error("获取 commit %s 的父 commit 失败", commit_hash)
            print(f"获取 commit {commit_hash} 的父 commit 失败，跳过 Parent 测试。")
            commit["performance_test_parent"] = []
        else:
            parent_hash = proc.stdout.strip()
            logging.info("父 commit: %s", parent_hash)
            print(f"父 commit: {parent_hash}")
            parent_existing = commit.get("performance_test_parent", [])
            subprocess.run("git reset --hard", shell=True)
            ret = subprocess.run(f"git checkout {parent_hash}", shell=True)
            if ret.returncode != 0:
                logging.error("切换到父 commit %s 失败", parent_hash)
                print(f"切换到父 commit {parent_hash} 失败。")
                commit["performance_test_parent"] = []
            else:
                time.sleep(1)
                logging.info("【Parent 状态】开始测试")
                print("【Parent 状态】开始测试……")
                parent_results = run_tests_for_state(repo_path, src_path, parent_existing)
                commit["performance_test_parent"] = parent_results

        # 每处理完一个 commit 立刻写回 json 文件，保存当前进度
        with open(test_result_file, "w", encoding="utf-8") as f:
            json.dump(commits, f, indent=4)
        logging.info("commit %s 处理完成，数据已写回 %s", commit_hash, test_result_file)
        print(f"commit {commit_hash} 处理完成，数据已写回。")

    logging.info("所有测试完成，最终结果保存在 %s", test_result_file)
    print("\n所有测试完成，结果已保存到：", test_result_file)
    
    # 恢复环境：例如切换到 master 分支，可根据实际需要调整
    subprocess.run("git reset --hard", shell=True)
    subprocess.run("git checkout master", shell=True)
    logging.info("恢复到 master 分支")

if __name__ == "__main__":
    main()
