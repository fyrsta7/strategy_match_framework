import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 输入文件路径以及输出文件路径，根据 config.root_path 配置
INPUT_FILE = os.path.join(config.root_path, "benchmark", "redis", "test_result.json")
OUTPUT_FILE = os.path.join(config.root_path, "benchmark", "redis", "comp_test_result.json")

def average_results(test_runs):
    """
    对输入的若干次测试结果求平均。
    每个测试结果为一个字典，形如：
        {
           "TEST_NAME": {
               "throughput": <float>,
               "latency": {
                   "avg": <float>,
                   "min": <float>,
                   "p50": <float>,
                   "p95": <float>,
                   "p99": <float>,
                   "max": <float>
               }
           },
           ...
        }
    返回的结果结构同样包含每个 TEST_NAME 下各项指标的平均值。
    """
    aggregated = {}  # { test_name: { "throughput": sum, "latency": { sub_key: sum, ... } } }
    counter = {}     # { test_name: { "throughput": count, "latency": { sub_key: count, ... } } }
    for run in test_runs:
        for test_name, data in run.items():
            if test_name not in aggregated:
                aggregated[test_name] = {}
                counter[test_name] = {}
            # 处理 throughput 部分
            if "throughput" in data:
                aggregated[test_name].setdefault("throughput", 0)
                counter[test_name].setdefault("throughput", 0)
                aggregated[test_name]["throughput"] += data["throughput"]
                counter[test_name]["throughput"] += 1
            # 处理 latency 部分
            if "latency" in data:
                aggregated[test_name].setdefault("latency", {})
                counter[test_name].setdefault("latency", {})
                for sub_key, sub_val in data["latency"].items():
                    aggregated[test_name]["latency"].setdefault(sub_key, 0)
                    counter[test_name]["latency"].setdefault(sub_key, 0)
                    aggregated[test_name]["latency"][sub_key] += sub_val
                    counter[test_name]["latency"][sub_key] += 1

    avg_result = {}
    for test_name in aggregated:
        avg_result[test_name] = {}
        # 计算 throughput 平均值
        if "throughput" in aggregated[test_name]:
            cnt = counter[test_name]["throughput"]
            if cnt > 0:
                avg_result[test_name]["throughput"] = aggregated[test_name]["throughput"] / cnt
        # 计算 latency 内各指标平均值
        if "latency" in aggregated[test_name]:
            avg_result[test_name]["latency"] = {}
            for sub_key in aggregated[test_name]["latency"]:
                cnt = counter[test_name]["latency"].get(sub_key, 0)
                if cnt > 0:
                    avg_result[test_name]["latency"][sub_key] = aggregated[test_name]["latency"][sub_key] / cnt
    return avg_result

def compute_percentage_diff(parent, child):
    """
    计算 parent 与 child 平均结果之间的百分比变化。
    返回的结果结构与 parent/child 类似，计算公式：
         ((child - parent) / parent) * 100
    当 parent 指标为 0 时返回 None（无法计算）。
    """
    diff = {}
    for test_name in child:
        if test_name not in parent:
            continue
        diff[test_name] = {}
        # 计算 throughput 百分比变化
        if "throughput" in child[test_name] and "throughput" in parent[test_name]:
            p_val = parent[test_name]["throughput"]
            c_val = child[test_name]["throughput"]
            if p_val == 0:
                diff[test_name]["throughput"] = None
            else:
                diff[test_name]["throughput"] = ((c_val - p_val) / p_val) * 100
        # 计算 latency 内各指标的百分比变化
        if "latency" in child[test_name] and "latency" in parent[test_name]:
            diff[test_name]["latency"] = {}
            for sub_key in child[test_name]["latency"]:
                if sub_key in parent[test_name]["latency"]:
                    p_val = parent[test_name]["latency"][sub_key]
                    c_val = child[test_name]["latency"][sub_key]
                    if p_val == 0:
                        diff[test_name]["latency"][sub_key] = None
                    else:
                        diff[test_name]["latency"][sub_key] = ((c_val - p_val) / p_val) * 100
    return diff

def main():
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            commits = json.load(f)
    except Exception as e:
        print(f"读取输入文件 {INPUT_FILE} 失败: {e}")
        sys.exit(1)

    overall_result = {}
    for commit in commits:
        commit_hash = commit.get("hash", "unknown")
        child_runs = commit.get("performance_test_child", [])
        parent_runs = commit.get("performance_test_parent", [])
        commit_data = {}

        # 对 child 测试结果计算平均值（如果有测试数据）
        if child_runs:
            child_avg = average_results(child_runs)
            commit_data["child_avg"] = child_avg
        # 对 parent 测试结果计算平均值（如果有测试数据）
        if parent_runs:
            parent_avg = average_results(parent_runs)
            commit_data["parent_avg"] = parent_avg
        # 如果同时存在 child 和 parent 测试数据，则计算变化百分比
        if child_runs and parent_runs:
            diff_percentage = compute_percentage_diff(parent_avg, child_avg)
            commit_data["difference_percentage"] = diff_percentage

        overall_result[commit_hash] = commit_data

    # 写入输出文件
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(overall_result, f, indent=4, ensure_ascii=False)
        print(f"结果已写入 {OUTPUT_FILE}")
    except Exception as e:
        print(f"写入输出文件 {OUTPUT_FILE} 失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
