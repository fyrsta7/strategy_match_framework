import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 定义输入输出文件路径
DIR_NAME = "partial_commit_40000"
FILE_NAME = "partial_commit_40000_dsv3"
INPUT_FILE_PATH = config.root_path + f"python/2-general_strategy/result/{DIR_NAME}/{FILE_NAME}.json"
OUTPUT_FILE_PATH = config.root_path + f"python/2-general_strategy/result/{DIR_NAME}/{FILE_NAME}_true.json"

def filter_generic_optimization_commits():
    # 确保输入文件存在
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Input file does not exist: {INPUT_FILE_PATH}")
        return False
    
    try:
        # 读取输入 JSON 文件
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            commits = json.load(f)
        
        # 筛选 is_generic_optimization_final 字段为 true 的 commit
        filtered_commits = [commit for commit in commits if commit.get("is_generic_optimization_final") is True]
        
        # 将筛选后的结果写入新的 JSON 文件
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(filtered_commits, f, ensure_ascii=False, indent=4)
        
        print(f"Successfully extracted {len(filtered_commits)} commits with is_generic_optimization_final=true")
        print(f"Output saved to: {OUTPUT_FILE_PATH}")
        return True
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    filter_generic_optimization_commits()