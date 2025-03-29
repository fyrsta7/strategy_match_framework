import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def process_json_file(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        commits = json.load(f)
    
    # 处理每个commit
    for commit in commits:
        # 将optimization_summary_1转换为字符串列表
        if "optimization_summary_1" in commit:
            commit["optimization_summary_1"] = [commit["optimization_summary_1"]]
        
        # 将is_generic_optimization_1转换为字符串列表
        if "is_generic_optimization_1" in commit:
            commit["is_generic_optimization_1"] = [commit["is_generic_optimization_1"]]
        
        # 删除指定字段
        fields_to_remove = [
            "optimization_summary_3",
            "optimization_summary_5",
            "is_generic_optimization_3",
            "is_generic_optimization_5"
        ]
        for field in fields_to_remove:
            if field in commit:
                del commit[field]
    
    # 保存修改后的内容回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(commits, f, indent=4, ensure_ascii=False)
    
    print(f"处理完成: {file_path}")

if __name__ == "__main__":
    # 构建文件路径
    file_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_50.json")
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        process_json_file(file_path)
    else:
        print(f"错误: 文件 {file_path} 不存在")