import json
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def update_commits_with_summaries():
    """
    将小文件(10000)中的commit优化信息复制到大文件(40000)中的相应commit
    """
    # 定义文件路径
    BASE_DIR = os.path.join(config.root_path, "python/2-general_strategy/result")
    LARGE_FILE = os.path.join(BASE_DIR, "partial_commit_40000", "partial_commit_40000.json")
    SMALL_FILE = os.path.join(BASE_DIR, "partial_commit_10000", "partial_commit_10000_dsv3.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "partial_commit_40000", "partial_commit_40000_dsv3.json")
    
    print("正在读取小文件(10000)...")
    with open(SMALL_FILE, 'r', encoding='utf-8') as f:
        small_commits = json.load(f)
    
    print("正在读取大文件(40000)...")
    with open(LARGE_FILE, 'r', encoding='utf-8') as f:
        large_commits = json.load(f)
    
    print(f"小文件中包含 {len(small_commits)} 个commit")
    print(f"大文件中包含 {len(large_commits)} 个commit")
    
    # 创建一个哈希表存储小文件中的每个commit，键为(hash, repository_name)
    small_commits_dict = {}
    for commit in small_commits:
        key = (commit["hash"], commit["repository_name"])
        small_commits_dict[key] = commit
    
    # 要复制的字段
    fields_to_copy = [
        "optimization_summary",
        "is_generic_optimization",
        "optimization_summary_final",
        "is_generic_optimization_final"
    ]
    
    # 统计计数器
    updated_count = 0
    
    # 遍历大文件中的每个commit
    print("正在更新大文件中的commit...")
    for i, large_commit in enumerate(tqdm(large_commits)):
        key = (large_commit["hash"], large_commit["repository_name"])
        
        # 检查commit是否存在于小文件中
        if key in small_commits_dict:
            small_commit = small_commits_dict[key]
            
            # 复制额外信息字段
            for field in fields_to_copy:
                if field in small_commit:
                    large_commits[i][field] = small_commit[field]
            
            updated_count += 1
    
    print(f"成功更新了 {updated_count} 个commit")
    
    # 保存更新后的大文件
    print("正在保存更新后的文件...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(large_commits, f, ensure_ascii=False, indent=4)
    
    print(f"更新完成！结果已保存到: {OUTPUT_FILE}")

if __name__ == "__main__":
    update_commits_with_summaries()