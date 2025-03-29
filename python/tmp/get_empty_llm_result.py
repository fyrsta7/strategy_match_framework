import json
import os
import config

def load_json(input_path):
    """
    读取JSON文件并返回数据。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到文件: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解码错误: {e}")
    
    if not isinstance(data, list):
        raise ValueError("JSON数据格式错误：预期是一个列表。")
    
    return data

def is_llm_average_child_empty(commit):
    """
    判断一个commit的'llm_average_child'字段是否为空。
    """
    llm_average_child = commit.get("llm_average_child", None)
    if llm_average_child is None:
        return True
    if isinstance(llm_average_child, dict) and not llm_average_child:
        return True
    return False

def analyze_commits(data):
    """
    分析commit数据，找到'llm_average_child'为空的commit。
    """
    empty_llm_commits = []
    
    for commit in data:
        if is_llm_average_child_empty(commit):
            commit_hash = commit.get("hash", "未知哈希值")
            empty_llm_commits.append(commit_hash)
    
    return empty_llm_commits

def save_results(output_path, empty_llm_commits):
    """
    将统计结果保存到指定的输出文件中。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for commit_hash in empty_llm_commits:
            f.write(f"{commit_hash}\n")
    
    print(f"统计结果已保存到 {output_path}")

def main():
    # 配置路径
    root_path = config.root_path  # 确保config模块中定义了root_path
    input_path = os.path.join(root_path, "result", "rocksdb", "llm", "bm_25", "comp_result_related_4_o1-mini.json")
    output_path = os.path.join(root_path, "result", "rocksdb", "llm", "bm_25", "commits_with_empty_llm_average_child.txt")  # 输出的文本文件路径
    
    try:
        data = load_json(input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return
    
    empty_llm_commits = analyze_commits(data)
    
    # 打印统计结果
    print("=== LLM Average Child 为空的 Commit 统计 ===")
    print(f"总提交数: {len(data)}")
    print(f"LLM Average Child 为空的提交数: {len(empty_llm_commits)}")
    print("LLM Average Child 为空的 Commit Hash 列表：")
    for commit_hash in empty_llm_commits:
        print(f"- {commit_hash}")
    
    # 保存结果到文件
    try:
        save_results(output_path, empty_llm_commits)
    except Exception as e:
        print(f"写入输出文件时发生错误: {e}")

if __name__ == "__main__":
    main()