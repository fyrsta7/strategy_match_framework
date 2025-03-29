import os
import json
from rank_bm25 import BM25Okapi
from gensim.utils import simple_preprocess
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

ROOT_PATH = config.root_path
JSON_FILE = os.path.join(ROOT_PATH, "all_is_opt_final.json")
OUTPUT_FILE = os.path.join(ROOT_PATH, "commit_similarity_bm25_19998_top20.json")
TOP_N = 20  # 设置要返回的最相似的 commit 数量。如果需要全部相似度，则设置为 None
IGNORE_SAME_REPO = True  # 是否忽略与同一个代码库中的 commit 计算相似度

def load_commits(json_file):
    """
    加载 all_one_func.json 文件并返回 commit 列表。
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        commits = json.load(f)
    return commits

def read_before_func(root_path, repository_name, commit_hash):
    """
    根据 repository_name 和 commit_hash 读取 before_func.txt 的内容。
    """
    func_path = os.path.join(root_path, 'knowledge_base', repository_name, 'modified_file', commit_hash, 'before_func.txt')
    if not os.path.exists(func_path):
        print(f"Warning: {func_path} does not exist.")
        return None
    with open(func_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def tokenize(text):
    """
    使用 gensim 的 simple_preprocess 进行简单分词。
    """
    return simple_preprocess(text, deacc=True)  # deacc=True 移除标点符号

def build_bm25_index(documents):
    """
    构建 BM25 索引。
    """
    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

def compute_bm25_similarity(bm25, query_tokens, top_n=None, query_repo=None, commit_map=None):
    """
    计算 BM25 相似度分数，并根据 IGNORE_SAME_REPO 筛选结果。
    """
    scores = bm25.get_scores(query_tokens)
    # 按分数降序排序
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # 如果忽略同一个仓库的 commit
    if IGNORE_SAME_REPO and query_repo is not None and commit_map is not None:
        sorted_scores = [
            (doc_idx, score) for doc_idx, score in sorted_scores
            if commit_map[doc_idx]["repository_name"] != query_repo  # 只保留不同仓库的 commit
        ]

    # 截断到 top_n
    if top_n:
        return sorted_scores[:top_n]
    else:
        return sorted_scores

def save_similarity_results(similarity_dict, commit_map, output_file):
    """
    保存相似度结果为 JSON 文件（已保证降序排列）
    """
    output_data = []
    for query_hash, similarities in similarity_dict.items():
        query_repo = query_map.get(query_hash, "Unknown")
        sim_list = []
        for doc_idx, score in similarities:
            doc_info = commit_map.get(doc_idx, {"commit_hash": "Unknown", "repository_name": "Unknown"})
            doc_hash = doc_info["commit_hash"]
            doc_repo = doc_info["repository_name"]
            if query_hash == doc_hash and query_repo == doc_repo:
                continue  # 排除与自身的相似度
            sim_list.append({
                "commit_hash": doc_hash,
                "repository_name": doc_repo,
                "similarity_score": float(score)  # 转换为Python原生float类型
            })
        
        # 最终确认排序（冗余校验）
        sim_list.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        output_data.append({
            "query_commit": {
                "commit_hash": query_hash,
                "repository_name": query_repo
            },
            "similar_commits": sim_list
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Similarity results saved to {output_file}")

def main():
    # 加载所有 commits
    commits = load_commits(JSON_FILE)
    print(f"Total commits loaded: {len(commits)}")

    # 构建索引映射
    commit_map = {idx: {"commit_hash": commit['hash'], "repository_name": commit['repository_name']} 
                 for idx, commit in enumerate(commits)}

    # 构建查询映射
    global query_map
    query_map = {commit['hash']: commit['repository_name'] for commit in commits}

    # 提取所有 before_func 内容
    all_funcs = []
    for commit in tqdm(commits, desc="Reading all before_func.txt"):
        content = read_before_func(ROOT_PATH, commit['repository_name'], commit['hash'])
        all_funcs.append(content if content else "")  # 空字符串占位

    # 构建 BM25 索引
    print("Building BM25 index...")
    bm25, tokenized_corpus = build_bm25_index(all_funcs)
    print("BM25 index built successfully.")

    # 筛选 rocksdb commits
    rocksdb_commits = [commit for commit in commits if commit['repository_name'] == 'rocksdb']
    print(f"Total rocksdb commits: {len(rocksdb_commits)}")

    # 准备 rocksdb 数据
    rocksdb_data = []
    for commit in tqdm(rocksdb_commits, desc="Processing rocksdb commits"):
        content = read_before_func(ROOT_PATH, commit['repository_name'], commit['hash'])
        rocksdb_data.append((
            commit['hash'],
            commit['repository_name'],
            content if content else ""
        ))

    # 计算相似度
    similarity_dict = {}
    for commit_hash, repo_name, func in tqdm(rocksdb_data, desc="Computing similarities"):
        if not func.strip():
            similarity_dict[commit_hash] = []
            continue
        query_tokens = tokenize(func)
        # 调用 compute_bm25_similarity 时传入 query_repo 和 commit_map
        similarities = compute_bm25_similarity(bm25, query_tokens, top_n=TOP_N, query_repo=repo_name, commit_map=commit_map)
        similarity_dict[commit_hash] = similarities

    # 保存结果
    save_similarity_results(similarity_dict, commit_map, OUTPUT_FILE)

if __name__ == "__main__":
    main()