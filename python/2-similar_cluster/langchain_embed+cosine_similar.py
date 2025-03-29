"""
示例代码，展示如何利用 embeddings+余弦相似度实现：
   1. 针对知识库中所有 commit 的 before_func.txt 进行 embedding 并构建向量库
   2. 对需要被优化的函数（例如来自某一特定代码库，如 "rocksdb"）计算 embedding，
      然后在向量层面计算与所有知识库函数之间的余張相似度
   3. 按相似度从高到低排序，筛选出 top_n 个相似例子，最终以 JSON 格式输出，格式与参考脚本一致
       
注意：
  1. 需要先安装依赖库：
       pip install -U langchain-openai numpy tqdm
  2. 请确保 config.py 文件中定义了：
          root_path            # 知识库根目录
          closeai_base_url     # 例如："https://api.openai.com/v1"
          closeai_api_key      # 替换为你的 API key
          closeai_chatgpt_model（如果需要后续调用 LLM，这里本脚本仅计算相似度）
  3. BM25 基于词频统计，不适用于 dense 向量相似度计算，本文使用余弦相似度进行计算。
  4. 为唯一确定一个 commit，使用仓库名和 commit_hash 的复合 key，格式为 "repository_name::commit_hash"
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings

# 添加项目根目录到 sys.path，以便载入 config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局路径和参数设置
ROOT_PATH = config.root_path
JSON_FILE = os.path.join(ROOT_PATH, "all_is_opt_final.json")
OUTPUT_FILE = os.path.join(ROOT_PATH, "commit_similarity_langchain_19998_top20.json")
TOP_N = 20  # 返回的最相似 commit 数量
IGNORE_SAME_REPO = True  # 是否忽略来自同一代码库的 commit

# 最大并行数量
MAX_WORKERS = 16

def load_commits(json_file):
    """
    加载 all_one_func.json 文件并返回 commits 列表
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        commits = json.load(f)
    return commits

def read_before_func(root_path, repository_name, commit_hash):
    """
    根据 repository_name 和 commit_hash 读取 before_func.txt 的内容
    """
    func_path = os.path.join(root_path, 'knowledge_base', repository_name, 'modified_file', commit_hash, 'before_func.txt')
    if not os.path.exists(func_path):
        print(f"Warning: {func_path} does not exist.")
        return None
    with open(func_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def cosine_similarity(vec_a, vec_b):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def save_similarity_results(similarity_dict, commit_map, output_file):
    """
    保存相似度结果到 JSON 文件，格式与参考脚本一致
    """
    output_data = []
    for query_key, similarities in similarity_dict.items():
        query_repo = query_map.get(query_key, "Unknown")
        # query_key 格式为 "repository::commit_hash"
        repo_name, commit_hash = query_key.split("::", 1)
        sim_list = []
        for doc_idx, score in similarities:
            doc_key = commit_map.get(doc_idx, {"composite_key": "Unknown"})
            # doc_key 中存储着 composite_key 格式为 "repository::commit_hash"
            composite_key = doc_key["composite_key"]
            # 排除自身
            if query_key == composite_key:
                continue
            d_repo, d_commit = composite_key.split("::", 1)
            sim_list.append({
                "commit_hash": d_commit,
                "repository_name": d_repo,
                "similarity_score": float(score)
            })
        sim_list.sort(key=lambda x: x["similarity_score"], reverse=True)
        output_data.append({
            "query_commit": {
                "commit_hash": commit_hash,
                "repository_name": query_repo
            },
            "similar_commits": sim_list
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Similarity results saved to {output_file}")

def main():
    # 加载所有 commit
    commits = load_commits(JSON_FILE)
    print(f"Total commits loaded: {len(commits)}")
    
    # 构建映射：commit_map 与 query_map 均使用复合 key "repository::commit_hash"
    # commit_map: 索引 idx -> {"composite_key": "repo::hash"}
    commit_map = {}
    # query_map: composite_key -> repository_name (用于输出)
    global query_map
    query_map = {}
    # commit_key_to_index: composite_key -> idx （用于快速定位 embedding）
    commit_key_to_index = {}
    
    for idx, commit in enumerate(commits):
        composite_key = f"{commit['repository_name']}::{commit['hash']}"
        commit_map[idx] = {"composite_key": composite_key}
        query_map[composite_key] = commit['repository_name']
        commit_key_to_index[composite_key] = idx
    
    # 提取所有 commit 的 before_func 内容
    all_funcs = []
    for commit in tqdm(commits, desc="Reading all before_func.txt"):
        content = read_before_func(ROOT_PATH, commit['repository_name'], commit['hash'])
        all_funcs.append(content if content else "")
    
    # 使用新版 embeddings 方法对所有函数计算 embedding，并构建向量库
    embeddings_model = OpenAIEmbeddings(model="closeai/text-embedding-ada-002", openai_api_key=config.xmcp_api_key_unlimit, openai_api_base=config.xmcp_base_url)
    # 利用 ThreadPoolExecutor 并行计算 embeddings
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        doc_embeddings = list(tqdm(executor.map(embeddings_model.embed_query, all_funcs),
                                     total=len(all_funcs),
                                     desc="Computing embeddings for all commits"))
    
    # 筛选出需要被优化的 commit，比如本例中 repository 为 "rocksdb" 的 commit 为查询
    rocksdb_commits = [commit for commit in commits if commit['repository_name'] == 'rocksdb']
    print(f"Total rocksdb commits: {len(rocksdb_commits)}")
    
    # 准备 rocksdb 数据，数据格式为 (composite_key, repository_name, before_func_content)
    rocksdb_data = []
    for commit in tqdm(rocksdb_commits, desc="Processing rocksdb commits"):
        composite_key = f"{commit['repository_name']}::{commit['hash']}"
        content = read_before_func(ROOT_PATH, commit['repository_name'], commit['hash'])
        rocksdb_data.append((
            composite_key,
            commit['repository_name'],
            content if content else ""
        ))
    
    # 对每个查询 commit（需要被优化的函数）计算与所有知识库中 commit 的余弦相似度，取 top_n 个
    similarity_dict = {}
    for query_key, repo_name, func in tqdm(rocksdb_data, desc="Computing similarities"):
        if not func.strip():
            similarity_dict[query_key] = []
            continue
        # 根据 composite key 查找查询 commit 的 embedding索引
        query_idx = commit_key_to_index.get(query_key, None)
        if query_idx is None:
            similarity_dict[query_key] = []
            continue
        
        query_embedding = doc_embeddings[query_idx]
        similarities = []
        # 遍历整个知识库，计算当前 query 与每个 commit 的相似度
        for idx, emb in enumerate(doc_embeddings):
            score = cosine_similarity(query_embedding, emb)
            # 如果设置忽略同一仓库，则跳过同 repo 的 commit（包括自身，后续会排除）
            # 从 commit_map 获取对应的 composite key，并拆分仓库名称
            comp_key = commit_map[idx]["composite_key"]
            other_repo = comp_key.split("::")[0]
            if IGNORE_SAME_REPO and other_repo == repo_name:
                continue
            similarities.append((idx, score))
        similarities.sort(key=lambda x: x[1], reverse=True)
        if TOP_N:
            similarities = similarities[:TOP_N]
        similarity_dict[query_key] = similarities
    
    # 保存相似度结果到 JSON 文件（格式与参考脚本相同）
    save_similarity_results(similarity_dict, commit_map, OUTPUT_FILE)

if __name__ == "__main__":
    main()
