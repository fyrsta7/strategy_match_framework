import json
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 文件路径
small_file_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_10000.json")
large_file_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_70330.json")
output_file_path = os.path.join(config.root_path, "python/2-general_strategy/result/merged_commit_70330.json")

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_commit_key(commit):
    """为commit创建唯一键值"""
    return f"{commit['hash']}_{commit['repository_name']}"

def process_chunk(args):
    """处理一个数据块"""
    chunk, commit_map = args
    updated_chunk = []
    
    for commit in chunk:
        key = create_commit_key(commit)
        if key in commit_map:
            # 如果当前commit在小文件中存在，使用小文件中的信息
            updated_chunk.append(commit_map[key])
        else:
            # 否则保持不变
            updated_chunk.append(commit)
    
    return updated_chunk

def main():
    print("Loading the smaller JSON file...")
    small_commits = load_json_file(small_file_path)
    
    print("Creating lookup map for smaller file commits...")
    # 为小文件中的每个commit创建一个字典，用于快速查找
    commit_map = {}
    for commit in small_commits:
        key = create_commit_key(commit)
        commit_map[key] = commit
    
    print("Loading the larger JSON file...")
    large_commits = load_json_file(large_file_path)
    
    print(f"Processing {len(large_commits)} commits with {multiprocessing.cpu_count()} cores...")
    
    # 根据CPU核心数量分割数据
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(large_commits) // num_cores
    if chunk_size == 0:
        chunk_size = 1
    
    chunks = [large_commits[i:i+chunk_size] for i in range(0, len(large_commits), chunk_size)]
    
    # 准备并行处理的参数
    process_args = [(chunk, commit_map) for chunk in chunks]
    
    # 并行处理
    updated_commits = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_chunk, process_args), total=len(chunks), desc="Processing chunks"))
        for result in results:
            updated_commits.extend(result)
    
    print(f"Writing {len(updated_commits)} processed commits to output file...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_commits, f, ensure_ascii=False, indent=2)
    
    print("Completed! Output saved to:", output_file_path)

if __name__ == "__main__":
    main()