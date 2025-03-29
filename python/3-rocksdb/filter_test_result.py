import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def filter_commits(input_path, output_path):
    """
    过滤JSON文件中的提交，仅保留那些average_parent字段不为空且包含完整数据的提交。

    :param input_path: 输入JSON文件的路径。
    :param output_path: 输出过滤后JSON文件的路径。
    """
    try:
        # 读取输入JSON文件
        with open(input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"错误: 未找到文件: {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"错误: JSON解码失败: {e}")
        return

    if not isinstance(data, list):
        print("错误: JSON数据格式错误，预期为一个列表。")
        return

    # 过滤提交：保留average_parent存在且包含fillseq和readrandom且这两个字段不为空的提交
    filtered_data = []
    removed_commits = 0
    for commit in data:
        average_parent = commit.get("average_parent", {})
        # 检查average_parent是否为字典，且包含'fillseq'和'readrandom'，且这两个字段都有有效数据
        if (
            isinstance(average_parent, dict) and
            "fillseq" in average_parent and isinstance(average_parent["fillseq"], dict) and len(average_parent["fillseq"]) > 0 and
            "readrandom" in average_parent and isinstance(average_parent["readrandom"], dict) and len(average_parent["readrandom"]) > 0
        ):
            filtered_data.append(commit)
        else:
            removed_commits += 1

    print(f"过滤完成: 保留了 {len(filtered_data)} 个提交，删除了 {removed_commits} 个提交。")

    # 写入输出JSON文件
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
        print(f"已成功将过滤后的数据写入 {output_path}")
    except Exception as e:
        print(f"错误: 写入JSON文件失败: {e}")

if __name__ == "__main__":
    # 定义输入和输出文件路径
    INPUT_JSON_PATH = config.root_path + "result/rocksdb/test_result.json"
    OUTPUT_JSON_PATH = config.root_path + "result/rocksdb/filtered_test_result.json"

    # 检查输入文件是否存在
    if not os.path.isfile(INPUT_JSON_PATH):
        print(f"错误: 输入文件不存在: {INPUT_JSON_PATH}")
    else:
        # 执行过滤操作
        filter_commits(INPUT_JSON_PATH, OUTPUT_JSON_PATH)