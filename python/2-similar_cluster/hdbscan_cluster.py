"""
HDBSCAN聚类实现:
从本地存储的模型中加载 commit message 与代码 diff 的向量，
批量处理 config.root_path/all_one_func_1870.json 中所有 commit 的信息，
计算融合向量、降维、HDBSCAN聚类，并将聚类结果按聚类标签分组后保存到 JSON 文件中。
"""
import os
import sys
import json
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import hdbscan
import numpy as np

# 设置环境变量，避免联网以及 fork 后 tokenizers 的并行警告，并限制线程数
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "64"  # 限制OpenBLAS线程数
os.environ["OMP_NUM_THREADS"] = "64"       # 限制OpenMP线程数
os.environ["MKL_NUM_THREADS"] = "64"       # 限制MKL线程数

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 降维后的维度数量
N_COMPONENTS = 50
# HDBSCAN 中每个聚类的最少commit数
MIN_CLUSTER_SIZE = 3
# commit 信息文件路径
COMMIT_FILE = os.path.join(config.root_path, "all_is_opt_final.json")

def get_code_embedding(code_str, max_length=512, device='cpu', tokenizer_code=None, model_code=None):
    """
    使用 CodeBERT 获取代码 diff 的向量表示，
    这里选用 [CLS] token 的向量作为整体表示。
    """
    inputs = tokenizer_code(code_str, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_code(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token vector
    return cls_embedding.squeeze(0)

def group_by_cluster(labels, commit_list):
    """
    根据聚类标签将 commit 列表分组，返回字典 {cluster_label: [commit1, commit2, ...]}。
    """
    clusters = {}
    for label, commit in zip(labels, commit_list):
        clusters.setdefault(str(label), []).append(commit)
    return clusters

def perform_hdbscan_clustering():
    # --------------------------
    # 1. 加载本地模型
    # --------------------------
    print("Loading local Sentence-BERT model from {}...".format(os.path.join(config.root_path, "models/all-MiniLM-L6-v2")))
    model_msg = SentenceTransformer(os.path.join(config.root_path, "models/all-MiniLM-L6-v2"))
    print("Loading local CodeBERT model from {}...".format(os.path.join(config.root_path, "models/codebert-base")))
    tokenizer_code = AutoTokenizer.from_pretrained(os.path.join(config.root_path, "models/codebert-base"))
    model_code = AutoModel.from_pretrained(os.path.join(config.root_path, "models/codebert-base"))

    # --------------------------
    # 2. 加载 commit 信息以及获取 diff 文本
    # --------------------------
    with open(COMMIT_FILE, "r", encoding="utf-8") as f:
        commits_data = json.load(f)
    commit_messages = []
    commit_diffs = []
    # 保留原始 commit 信息以便后续输出
    commit_infos = []
    print("Loading commit messages and reading diff files...")
    for commit in tqdm(commits_data, desc="Processing commits"):
        # 获取 commit message
        message = commit.get("message", "")
        commit_messages.append(message)
        commit_infos.append(commit)
        # 构造 diff 文件路径
        repo_name = commit.get("repository_name", "")
        commit_hash = commit.get("hash", "")
        diff_path = os.path.join(config.root_path, "result", repo_name, "modified_file", commit_hash, "diff.txt")
        if os.path.exists(diff_path):
            try:
                with open(diff_path, "r", encoding="utf-8") as df:
                    diff_text = df.read()
            except Exception as e:
                print("Error reading diff file {}: {}".format(diff_path, e))
                diff_text = ""
        else:
            diff_text = ""
        commit_diffs.append(diff_text)
    N_total = len(commit_messages)
    print("Total commits to process: {}".format(N_total))

    # --------------------------
    # 3. 设备判断（优先使用 GPU，否则 CPU）
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_msg.to(device)
    model_code.to(device)

    # --------------------------
    # 4. 提取向量（批量处理支持进度显示）
    # --------------------------
    print("Extracting commit message embeddings (batch processing)...")
    embeddings_msg = model_msg.encode(commit_messages, convert_to_tensor=True, device=device)
    print("Extracting code diff embeddings...")
    embeddings_code_list = []
    for diff_text in tqdm(commit_diffs, desc="Processing diffs"):
        emb = get_code_embedding(diff_text, device=device, tokenizer_code=tokenizer_code, model_code=model_code)
        embeddings_code_list.append(emb)
    embeddings_code = torch.stack(embeddings_code_list)

    # --------------------------
    # 5. 融合两个向量
    # --------------------------
    print("Fusing message and code embeddings...")
    combined_embeddings = torch.cat([embeddings_msg, embeddings_code], dim=1)
    combined_embeddings_np = combined_embeddings.cpu().numpy()

    # --------------------------
    # 6. 降维
    # --------------------------
    print("Performing PCA for dimensionality reduction (n_components = {})...".format(N_COMPONENTS))
    # 使用批处理进行PCA以减少内存使用
    batch_size = 5000  # 调整这个值以适应您的内存
    total_samples = combined_embeddings_np.shape[0]
    if total_samples > batch_size:
        print(f"Using batch processing for PCA with batch size {batch_size}")
        # 先用小批量拟合PCA模型
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        pca.fit(combined_embeddings_np[:min(batch_size*2, total_samples)])
        # 然后分批转换数据
        embeddings_reduced = []
        for i in range(0, total_samples, batch_size):
            end = min(i + batch_size, total_samples)
            batch = combined_embeddings_np[i:end]
            reduced_batch = pca.transform(batch)
            embeddings_reduced.append(reduced_batch)
        embeddings_reduced = np.vstack(embeddings_reduced)
    else:
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        embeddings_reduced = pca.fit_transform(combined_embeddings_np)

    # --------------------------
    # 7. HDBSCAN聚类
    # --------------------------
    print("Clustering using HDBSCAN...")
    # 对HDBSCAN使用默认值，或者考虑降低大小以减少内存使用
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, core_dist_n_jobs=1)  # 限制并行作业数
    cluster_labels_hdbscan = clusterer.fit_predict(embeddings_reduced)

    # --------------------------
    # 8. 输出聚类统计信息
    # --------------------------
    distinct_hdbscan_labels = set(cluster_labels_hdbscan)
    num_clusters_hdbscan = len([lab for lab in distinct_hdbscan_labels if lab != -1])
    num_noise = sum(1 for lab in cluster_labels_hdbscan if lab == -1)
    num_assigned = N_total - num_noise
    print("HDBSCAN Clustering: Obtained {} clusters (excluding noise).".format(num_clusters_hdbscan))
    print("HDBSCAN: {} points assigned to clusters, {} points identified as noise.".format(num_assigned, num_noise))

    # --------------------------
    # 9. 整理聚类结果（按聚类标签分组）
    # --------------------------
    clusters_hdbscan = group_by_cluster(cluster_labels_hdbscan, commit_infos)

    # --------------------------
    # 10. 保存聚类结果到 JSON 文件
    # --------------------------
    cluster_result_hdbscan_path = os.path.join(config.root_path, f"cluster_result_hdbscan_{N_COMPONENTS}_{MIN_CLUSTER_SIZE}.json")
    print("Saving HDBSCAN clustering result to {}...".format(cluster_result_hdbscan_path))
    with open(cluster_result_hdbscan_path, "w", encoding="utf-8") as f:
        json.dump(clusters_hdbscan, f, ensure_ascii=False, indent=4)
    
    print("\nHDBSCAN processing complete. Total commits processed: {}.".format(N_total))
    return clusters_hdbscan

if __name__ == "__main__":
    perform_hdbscan_clustering()