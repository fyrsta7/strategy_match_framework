import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局变量：设置提取最相似 commit 的数量
NUM_SIMILAR = 4

# 输入文件：commit_similarity_langchain+cosine.json 位于 config.root_path 下
SIMILARITY_FILE = os.path.join(config.root_path, 'commit_similarity_langchain+cosine.json')

# 输出文件：汇总结果保存到 benchmark/rocksdb/related_commit_langchain+cosine.json
OUTPUT_FILE = os.path.join(config.root_path, 'benchmark', 'rocksdb', 'related_commit_langchain+cosine.json')

def get_top_unique_commits(similar_commits, top_n):
    """
    从 similar_commits 中挑选出 top_n 个相似度最高且唯一（由 repository_name 与 commit_hash 组成）的 commit。
    如果出现重复，则用后面相似度更低的 commit 来补全结果。
    """
    top_unique = []
    seen = set()
    # 先根据 similarity_score 降序排序
    similar_sorted = sorted(similar_commits, key=lambda x: x.get("similarity_score", 0), reverse=True)
    for commit in similar_sorted:
        repo = commit.get("repository_name")
        commit_hash = commit.get("commit_hash")
        # 构造唯一标识符
        key = (repo, commit_hash)
        if key in seen:
            continue  # 如果已经存在则跳过
        top_unique.append(commit)
        seen.add(key)
        if len(top_unique) == top_n:
            break
            
    return top_unique

def process_similarity_data():
    """
    从 commit_similarity.json 加载数据，
    针对每个 query_commit，提取其最相似的 NUM_SIMILAR 个唯一 commit 信息，
    将结果汇总后写入到 OUTPUT_FILE 指定的文件中。
    """
    # 读取相似性数据
    try:
        with open(SIMILARITY_FILE, 'r', encoding='utf-8') as f:
            similarity_data = json.load(f)
    except Exception as e:
        print(f"错误：无法读取相似性数据文件 {SIMILARITY_FILE}。{e}")
        sys.exit(1)
    
    aggregated_results = []
    
    for entry in similarity_data:
        query_commit = entry.get("query_commit")
        if not query_commit:
            continue
        
        # 获取该 commit 的相似 commit 列表（如果不存在，则默认为空列表）
        similar_commits = entry.get("similar_commits", [])
        # 提取 NUM_SIMILAR 个唯一 commit
        top_unique_commits = get_top_unique_commits(similar_commits, NUM_SIMILAR)
        
        aggregated_results.append({
            "query_commit": query_commit,
            "top_similar_commits": top_unique_commits
        })
    
    # 保证输出文件所在目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(aggregated_results, f, indent=4)
        print(f"汇总后的相似 commit 信息已保存到 {OUTPUT_FILE}")
    except Exception as e:
        print(f"错误：无法写入输出文件 {OUTPUT_FILE}。{e}")
        sys.exit(1)

if __name__ == "__main__":
    process_similarity_data()
