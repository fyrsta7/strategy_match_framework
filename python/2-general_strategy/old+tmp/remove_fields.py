import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

file_path = config.root_path + "python/2-general_strategy/result/partial_commit_test.json"

def remove_fields_from_commits(file_path):
    # 读取JSON文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 需要删除的字段列表
    fields_to_remove = [
        "optimization_summary",
        "is_generic_optimization",
        "optimization_summary_final",
        "is_generic_optimization_final"
    ]
    
    # 从每个commit记录中删除指定字段
    for commit in commits:
        for field in fields_to_remove:
            if field in commit:
                del commit[field]
    
    # 将修改后的数据写回文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(commits, f, indent=4, ensure_ascii=False)
        print(f"已成功从 {len(commits)} 个提交记录中删除指定字段。")
    except Exception as e:
        print(f"写入文件时出错: {e}")

if __name__ == "__main__":
    remove_fields_from_commits(file_path)