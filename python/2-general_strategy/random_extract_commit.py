import json
import random
import os
from typing import List, Dict, Any
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局变量配置
# 提取的 commit 总数
N_COMMITS = 40000
# 输入输出文件路径
INPUT_FILE = os.path.join(config.root_path, "all_is_opt_final.json")
BASE_DIR = os.path.join(config.root_path, "python/2-general_strategy/result")
OUTPUT_FILE = os.path.join(BASE_DIR, f"partial_commit_{N_COMMITS}", f"partial_commit_{N_COMMITS}.json")

# 是否从额外文件导入commits
USE_EXTRA_INPUT = True
# 额外输入文件路径
EXTRA_INPUT_FILE = os.path.join(BASE_DIR, "partial_commit_10000", "partial_commit_10000.json")

def extract_partial_commits():
    """
    从知识库中提取所有rocksdb的commit和随机选择的其他commit
    """
    # 创建输出目录(如果不存在)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 用于存储最终选择的commits
    final_commits = []
    # 用于存储额外输入文件中的commit的唯一标识符(hash+repo)，用于后续去重
    extra_commit_identifiers = set()
    
    # 如果启用了额外输入文件
    if USE_EXTRA_INPUT and os.path.exists(EXTRA_INPUT_FILE):
        print(f"正在从额外输入文件 {EXTRA_INPUT_FILE} 读取commits...")
        with open(EXTRA_INPUT_FILE, 'r', encoding='utf-8') as f:
            extra_commits = json.load(f)
        print(f"从额外文件读取到 {len(extra_commits)} 个commit")
        
        # 将额外文件中的commits添加到最终结果中
        final_commits = extra_commits.copy()
        
        # 记录额外文件中所有commit的唯一标识符
        for commit in extra_commits:
            commit_id = (commit["hash"], commit["repository_name"])
            extra_commit_identifiers.add(commit_id)
    
    print("正在读取所有commit数据...")
    # 读取所有commit
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_commits = json.load(f)
    
    print(f"总共读取到 {len(all_commits)} 个commit")
    
    # 将commits分为rocksdb和非rocksdb两组，同时排除额外输入文件中的commit
    rocksdb_commits = []
    other_commits = []
    
    print("正在分类commit...")
    for commit in tqdm(all_commits, desc="分类commits"):
        commit_id = (commit["hash"], commit["repository_name"])
        # 如果这个commit不在额外输入文件中，才考虑它
        if commit_id not in extra_commit_identifiers:
            if commit["repository_name"] == "rocksdb":
                rocksdb_commits.append(commit)
            else:
                other_commits.append(commit)
    
    print(f"找到 {len(rocksdb_commits)} 个新的rocksdb的commit")
    print(f"找到 {len(other_commits)} 个新的非rocksdb的commit")
    
    # 创建一个函数来检查commit是否在最终列表中
    def is_in_final_list(commit, commit_list):
        for existing_commit in commit_list:
            if (commit["hash"] == existing_commit["hash"] and 
                commit["repository_name"] == existing_commit["repository_name"]):
                return True
        return False
    
    # 检查最终结果中已有的rocksdb commits
    existing_rocksdb_commits = [c for c in final_commits if c["repository_name"] == "rocksdb"]
    print(f"最终结果中已有 {len(existing_rocksdb_commits)} 个rocksdb的commit")
    
    # 添加rocksdb commits直到达到50个
    needed_rocksdb = max(0, 50 - len(existing_rocksdb_commits))
    added_rocksdb = 0
    
    if needed_rocksdb > 0:
        print(f"尝试添加 {needed_rocksdb} 个rocksdb的commit...")
        for commit in rocksdb_commits:
            if not is_in_final_list(commit, final_commits):
                final_commits.append(commit)
                added_rocksdb += 1
                if added_rocksdb >= needed_rocksdb:
                    break
        
        print(f"成功添加 {added_rocksdb} 个新的rocksdb commit")
    
    # 计算还需要多少commit
    needed_total = max(0, N_COMMITS - len(final_commits))
    print(f"需要再添加 {needed_total} 个commit以达到目标数量 {N_COMMITS}")
    
    # 如果还需要添加其他commit
    if needed_total > 0:
        if needed_total > len(other_commits):
            print(f"警告: 请求的额外commit数量 ({needed_total}) 超过了可用的数量 ({len(other_commits)})")
            needed_total = len(other_commits)
        
        print(f"正在随机抽取 {needed_total} 个额外的commit...")
        # 随机抽取其他commit
        selected_other_commits = random.sample(other_commits, needed_total)
        
        # 添加到最终结果
        final_commits.extend(selected_other_commits)
    
    print("正在保存结果...")
    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_commits, f, ensure_ascii=False, indent=4)
    
    # 计算最终结果中rocksdb和非rocksdb的数量
    final_rocksdb_count = sum(1 for commit in final_commits if commit["repository_name"] == "rocksdb")
    final_other_count = len(final_commits) - final_rocksdb_count
    
    print(f"成功提取总计 {len(final_commits)} 个commit，其中包含:")
    print(f"- {final_rocksdb_count} 个rocksdb commit")
    print(f"- {final_other_count} 个非rocksdb commit")
    print(f"结果已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_partial_commits()