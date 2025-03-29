import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def split_repo_list(input_file, first_file, middle_file, remaining_file):
    # 读取 json 文件中的数据，假设整个文件是一个列表
    with open(input_file, "r", encoding="utf-8") as f:
        repo_list = json.load(f)
    
    total = len(repo_list)
    print(f"总共有 {total} 条记录.")
    
    # 获取前 10,000 条记录
    first_part = repo_list[:10000]
    
    # 获取中间 10,000 条记录
    # 如果总数不足 20,000，则中间部分记录可能不足 10,000 条
    middle_part = repo_list[10000:20000]
    
    # 获取剩余的所有记录
    remaining_part = repo_list[20000:]
    
    # 保存到对应的文件中
    with open(first_file, "w", encoding="utf-8") as f:
        json.dump(first_part, f, indent=4, ensure_ascii=False)
    print(f"前 1w 条记录已保存到: {first_file}")
    
    with open(middle_file, "w", encoding="utf-8") as f:
        json.dump(middle_part, f, indent=4, ensure_ascii=False)
    print(f"中间 1w 条记录已保存到: {middle_file}")
    
    with open(remaining_file, "w", encoding="utf-8") as f:
        json.dump(remaining_part, f, indent=4, ensure_ascii=False)
    print(f"剩余记录已保存到: {remaining_file}")

if __name__ == "__main__":
    input_json = config.root_path + "repo_list_30345.json"
    output_first = config.root_path + "repo_list_30345_1.json"
    output_middle = config.root_path + "repo_list_30345_2.json"
    output_remaining = config.root_path + "repo_list_30345_3.json"
    
    split_repo_list(input_json, output_first, output_middle, output_remaining)
