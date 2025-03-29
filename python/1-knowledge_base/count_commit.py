import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def count_commits_in_benchmark():
    """
    计算 all_final_benchmark.json 文件中一共有多少个 commit
    """
    benchmark_file_path = os.path.join(config.root_path, "all_is_opt_final.json")
    # 检查文件是否存在
    if not os.path.exists(benchmark_file_path):
        print(f"文件 {benchmark_file_path} 不存在，请检查路径。")
        return

    try:
        # 读取 JSON 文件
        with open(benchmark_file_path, "r") as f:
            benchmark_data = json.load(f)

        # 确保数据是一个列表
        if not isinstance(benchmark_data, list):
            print("all_is_opt_final_has_duplicate.json 文件格式错误，预期为包含 commit 的列表。")
            return

        # 计算 commit 数量
        commit_count = len(benchmark_data)

        print(f"all_is_opt_final_has_duplicate.json 文件中一共有 {commit_count} 个 commit。")
        return commit_count

    except Exception as e:
        print(f"读取或解析文件时发生错误: {e}")


if __name__ == "__main__":
    count_commits_in_benchmark()