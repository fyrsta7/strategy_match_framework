import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import random

# 配置路径
CONFIG_ROOT_PATH = config.root_path
model = "gpt-4o"
# model = "o1-mini"
# model = "o3-mini"
# model = "deepseek-v3"
# model = "deepseek-reasoner"

JSON_DIR = CONFIG_ROOT_PATH + "benchmark/rocksdb/llm/baseline/"
INPUT_JSON_FILE_NAME = f"baseline_{model}_2.json"
INPUT_JSON_PATH = JSON_DIR + INPUT_JSON_FILE_NAME
OUTPUT_JSON_PATH = JSON_DIR + "comp_" + INPUT_JSON_FILE_NAME

def compare_values(parent_val, new_val, metric):
    """
    比较两个值，返回性能改进的结论 ('better', 'worse', 'no change')。
    """
    try:
        parent_num = float(parent_val)
        new_num = float(new_val)

        if metric in ["micros_per_op", "seconds"]:
            # 越小越好
            if new_num < parent_num:
                return "better"
            elif new_num > parent_num:
                return "worse"
            else:
                return "no change"
        elif metric in ["ops_per_sec", "MB_per_sec"]:
            # 越大越好
            if new_num > parent_num:
                return "better"
            elif new_num < parent_num:
                return "worse"
            else:
                return "no change"
        else:
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"

def compare_performance(parent_metrics, child_metrics):
    """
    比较 parent 和 child 的性能指标。
    返回一个字典形式的比较结果（基于 fillseq 和 readrandom）。
    仅比较存在于 parent 和 child 中的字段。
    """
    relevant_sections = ["fillseq", "readrandom"]
    comparison_result = {}

    for section in relevant_sections:
        if section in parent_metrics and section in child_metrics:
            better_count = 0
            worse_count = 0
            # 获取所有在 parent 和 child 中都存在的键
            common_keys = set(parent_metrics[section].keys()).intersection(child_metrics[section].keys())
            for key in common_keys:
                result = compare_values(parent_metrics[section][key], child_metrics[section][key], key)
                if result == "better":
                    better_count += 1
                elif result == "worse":
                    worse_count += 1
                # "no change" 和 "N/A" 不计入统计

            # 根据多数结果确定总体结论
            if better_count > worse_count:
                comparison_result[section] = "better"
            elif worse_count > better_count:
                comparison_result[section] = "worse"
            elif better_count == worse_count and better_count > 0:
                # 当数量相等且至少有一个"better"或"worse"，随机选择
                comparison_result[section] = random.choice(["better", "worse"])
            else:
                # 如果没有"better"或"worse"，则标记为"no change"
                comparison_result[section] = "no change"

    return comparison_result

def initialize_nested_distribution():
    """
    初始化嵌套的 Perf_Comp_Result_Distribution 统计结构。
    """
    return {
        "both_better": 0,
        "one_better_one_worse": 0,
        "both_worse": 0
    }

def update_origin_stats(stats, origin_result):
    """
    更新 origin_perf_comp_result 的统计数据。
    """
    if origin_result:
        fillseq_result = origin_result.get("fillseq")
        readrandom_result = origin_result.get("readrandom")

        if fillseq_result == "better" and readrandom_result == "better":
            stats["both_better"] += 1
        elif (fillseq_result == "better" and readrandom_result == "worse") or \
             (fillseq_result == "worse" and readrandom_result == "better"):
            stats["one_better_one_worse"] += 1
        elif fillseq_result == "worse" and readrandom_result == "worse":
            stats["both_worse"] += 1

def main():
    # 读取JSON数据
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"未找到文件: {INPUT_JSON_PATH}")
        return
    except json.JSONDecodeError as e:
        print(f"JSON解码错误: {e}")
        return

    if not isinstance(data, list):
        print("JSON数据格式错误：预期是一个列表。")
        return

    output_data = []
    stats = {
        # 统计所有提交的 parent_to_child
        "parent_to_child": {
            "both_better": 0,
            "one_better_one_worse": 0,
            "both_worse": 0
        },
        # 统计有 LLM 结果的提交上的 parent_to_child_with_llm
        "parent_to_child_with_llm": {
            "both_better": 0,
            "one_better_one_worse": 0,
            "both_worse": 0
        },
        # 针对 parent_to_llm 中的每个类别统计 parent_to_child 分布
        "parent_to_llm_distribution": {
            "both_better": initialize_nested_distribution(),
            "one_better_one_worse": initialize_nested_distribution(),
            "both_worse": initialize_nested_distribution()
        },
        # LLM 性能比较结果统计（parent_to_llm）
        "parent_to_llm": {
            "both_better": 0,
            "one_better_one_worse": 0,
            "both_worse": 0
        },
        # 新增: LLM 与 average_child 比较的统计 (child_to_llm)
        "child_to_llm": {
            "both_better": 0,
            "one_better_one_worse": 0,
            "both_worse": 0
        },
        # 新增: 针对 child_to_llm 结果中的每个类别统计 parent_to_child 分布
        "child_to_llm_distribution": {
            "both_better": initialize_nested_distribution(),
            "one_better_one_worse": initialize_nested_distribution(),
            "both_worse": initialize_nested_distribution()
        },
        # 新增: 存储llm_average_child为空的commit hashes
        "commits_with_empty_llm_average_child": []
    }

    for entry in data:
        # 初始化结果条目并复制所需字段
        result_entry = {
            "hash": entry.get("hash", ""),
            "can_compile_and_run": bool(entry.get("llm_performance_test_child")),
            "parent_to_child": None,
            "parent_to_llm": None,
            "child_to_llm": None,  # 新增字段
            "parent_to_child_with_llm": None,  # 新增字段
            "parent_to_llm_distribution": None,  # 新增字段
            "child_to_llm_distribution": None,  # 新增字段
            "average_parent": entry.get("average_parent", {}),
            "average_child": entry.get("average_child", {}),
            "llm_average_child": entry.get("llm_average_child", {}),
        }

        # 检查 llm_average_child 是否为空，如果为空，则将hash添加到stats中
        llm_average_child = entry.get("llm_average_child", {})
        if not llm_average_child:
            commit_hash = entry.get("hash", "未知哈希值")
            stats["commits_with_empty_llm_average_child"].append(commit_hash)

        # 比较 average_child 和 average_parent（parent_to_child）
        average_parent = entry.get("average_parent", {})
        average_child = entry.get("average_child", {})
        if average_parent and average_child:
            parent_to_child_result = compare_performance(average_parent, average_child)
            result_entry["parent_to_child"] = parent_to_child_result

            # 更新统计：统计所有提交上的 parent_to_child
            update_origin_stats(stats["parent_to_child"], parent_to_child_result)

        # 比较 llm_average_child 和 average_parent（parent_to_llm）
        if average_parent and llm_average_child:
            parent_to_llm_result = compare_performance(average_parent, llm_average_child)
            result_entry["parent_to_llm"] = parent_to_llm_result

            # 更新统计：统计 parent_to_llm
            if parent_to_llm_result:
                fillseq_result = parent_to_llm_result.get("fillseq")
                readrandom_result = parent_to_llm_result.get("readrandom")

                # 确定 parent_to_llm 比较的类别
                if fillseq_result == "better" and readrandom_result == "better":
                    category = "both_better"
                elif (fillseq_result == "better" and readrandom_result == "worse") or \
                     (fillseq_result == "worse" and readrandom_result == "better"):
                    category = "one_better_one_worse"
                elif fillseq_result == "worse" and readrandom_result == "worse":
                    category = "both_worse"
                else:
                    category = None  # 由于已经修改 compare_performance，不会出现"inconsistent"

                if category:
                    stats["parent_to_llm"][category] += 1

                    # 更新 parent_to_llm_distribution
                    if result_entry["parent_to_child"]:
                        origin_result = result_entry["parent_to_child"]
                        origin_fillseq = origin_result.get("fillseq", "no change")
                        origin_readrandom = origin_result.get("readrandom", "no change")

                        if origin_fillseq == "better" and origin_readrandom == "better":
                            stats["parent_to_llm_distribution"][category]["both_better"] += 1
                        elif (origin_fillseq == "better" and origin_readrandom == "worse") or \
                             (origin_fillseq == "worse" and origin_readrandom == "better"):
                            stats["parent_to_llm_distribution"][category]["one_better_one_worse"] += 1
                        elif origin_fillseq == "worse" and origin_readrandom == "worse":
                            stats["parent_to_llm_distribution"][category]["both_worse"] += 1

        # 比较 llm_average_child 和 average_child（child_to_llm）
        if average_child and llm_average_child:
            child_to_llm_result = compare_performance(average_child, llm_average_child)
            result_entry["child_to_llm"] = child_to_llm_result

            # 更新统计：统计 child_to_llm
            if child_to_llm_result:
                fillseq_result = child_to_llm_result.get("fillseq")
                readrandom_result = child_to_llm_result.get("readrandom")

                # 确定 child_to_llm 比较的类别
                if fillseq_result == "better" and readrandom_result == "better":
                    category = "both_better"
                elif (fillseq_result == "better" and readrandom_result == "worse") or \
                     (fillseq_result == "worse" and readrandom_result == "better"):
                    category = "one_better_one_worse"
                elif fillseq_result == "worse" and readrandom_result == "worse":
                    category = "both_worse"
                else:
                    category = None  # 由于已经修改 compare_performance，不会出现"inconsistent"

                if category:
                    stats["child_to_llm"][category] += 1

                    # 更新 child_to_llm_distribution
                    if result_entry["parent_to_child"]:
                        origin_result = result_entry["parent_to_child"]
                        origin_fillseq = origin_result.get("fillseq", "no change")
                        origin_readrandom = origin_result.get("readrandom", "no change")

                        if origin_fillseq == "better" and origin_readrandom == "better":
                            stats["child_to_llm_distribution"][category]["both_better"] += 1
                        elif (origin_fillseq == "better" and origin_readrandom == "worse") or \
                             (origin_fillseq == "worse" and origin_readrandom == "better"):
                            stats["child_to_llm_distribution"][category]["one_better_one_worse"] += 1
                        elif origin_fillseq == "worse" and origin_readrandom == "worse":
                            stats["child_to_llm_distribution"][category]["both_worse"] += 1

        # 比较 llm_average_child 和 average_parent 的变化，进一步记录有 LLM 的情况
        if average_parent and llm_average_child and average_child:
            # parent_to_child_with_llm
            if average_parent and average_child:
                if average_parent and average_child:
                    parent_to_child_result = compare_performance(average_parent, average_child)
                    if parent_to_child_result:
                        result_entry["parent_to_child_with_llm"] = parent_to_child_result

                        # 更新统计：统计有 LLM 结果的提交上的 parent_to_child_with_llm
                        update_origin_stats(stats["parent_to_child_with_llm"], parent_to_child_result)

        output_data.append(result_entry)

    # 调整 stats 字段顺序，并添加新的commit hashes列表
    stats_ordered = {
        "parent_to_child": stats["parent_to_child"],
        "parent_to_child_with_llm": stats["parent_to_child_with_llm"],
        "parent_to_llm": stats["parent_to_llm"],
        "child_to_llm": stats["child_to_llm"],
        "parent_to_llm_distribution": stats["parent_to_llm_distribution"],
        "child_to_llm_distribution": stats["child_to_llm_distribution"],
        "commits_with_empty_llm_average_child": stats["commits_with_empty_llm_average_child"]  # 新增字段
    }

    # 将统计数据添加到输出的 JSON 文件中
    output_data.append({"stats": stats_ordered})

    # 检查统计数据的一致性
    total_parent_to_llm = sum(stats["parent_to_llm"].values())
    total_parent_to_child_with_llm = sum(stats["parent_to_child_with_llm"].values())

    if total_parent_to_llm != total_parent_to_child_with_llm:
        print("警告：'parent_to_child_with_llm' 的总和与 'parent_to_llm' 的总和不一致。")
        print(f"'parent_to_child_with_llm' 总和: {total_parent_to_child_with_llm}")
        print(f"'parent_to_llm' 总和: {total_parent_to_llm}")
    else:
        print("统计数据一致：'parent_to_child_with_llm' 的总和与 'parent_to_llm' 的总和相等。")

    # 检查 child_to_llm 统计一致性
    total_child_to_llm = sum(stats["child_to_llm"].values())
    total_child_to_llm_distribution = sum(
        stats["child_to_llm_distribution"]["both_better"].values()) + \
        sum(stats["child_to_llm_distribution"]["one_better_one_worse"].values()) + \
        sum(stats["child_to_llm_distribution"]["both_worse"].values())

    if total_child_to_llm != total_child_to_llm_distribution:
        print("警告：'child_to_llm' 的总和与 'child_to_llm_distribution' 的总和不一致。")
        print(f"'child_to_llm' 总和: {total_child_to_llm}")
        print(f"'child_to_llm_distribution' 总和: {total_child_to_llm_distribution}")
    else:
        print("统计数据一致：'child_to_llm' 的总和与 'child_to_llm_distribution' 的总和相等。")

    # 检查 parent_to_llm_distribution 统计一致性
    total_parent_to_llm_distribution = sum(
        stats["parent_to_llm_distribution"]["both_better"].values()) + \
        sum(stats["parent_to_llm_distribution"]["one_better_one_worse"].values()) + \
        sum(stats["parent_to_llm_distribution"]["both_worse"].values())

    if total_parent_to_llm != total_parent_to_llm_distribution:
        print("警告：'parent_to_child_with_llm' 的总和与 'parent_to_llm_distribution' 的总和不一致。")
        print(f"'parent_to_child_with_llm' 总和: {total_parent_to_llm}")
        print(f"'parent_to_llm_distribution' 总和: {total_parent_to_llm_distribution}")
    else:
        print("统计数据一致：'parent_to_child_with_llm' 的总和与 'parent_to_llm_distribution' 的总和相等。")

    # 将结果写入新的 JSON 文件
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"数据已成功写入 {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"写入JSON文件时发生错误: {e}")

    # 打印空llm_average_child的commit hashes
    print("\n=== LLM Average Child 为空的 Commit Hash 列表 ===")
    for commit_hash in stats["commits_with_empty_llm_average_child"]:
        print(f"- {commit_hash}")

if __name__ == "__main__":
    main()