import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import copy
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 设置多进程启动方法为'spawn'
multiprocessing.set_start_method('spawn', force=True)

# 全局变量定义
MODEL_PATH = os.path.join(config.root_path, "models/all-MiniLM-L6-v2")
DIR_NAME = "kmeans_40000_dsv3_true_14000"
FILE_NAME = "init"
INPUT_FILE = os.path.join(config.root_path, f"python/2-general_strategy/result/{DIR_NAME}/{FILE_NAME}.json")
OUTPUT_FILE = os.path.join(config.root_path, f"python/2-general_strategy/result/{DIR_NAME}/order.json")
NUM_WORKERS = 64

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# 辅助函数，用于按 authority_score 降序排序
def sort_by_authority(commit):
    return commit["authority_score"]

def calculate_similarity_scores(commits, model_path):
    """
    计算每个commit的策略描述与其他commit策略描述的相似度
    并返回平均相似度作为该commit在聚类中的权威度分数
    """
    # 在每个工作进程中加载模型并明确设置使用CPU
    model = SentenceTransformer(model_path, device="cpu")
    
    # 提取所有策略描述
    strategy_descriptions = [commit["optimization_summary_final"] for commit in commits]
    
    # 使用sentence transformer将描述转换为向量
    embeddings = model.encode(strategy_descriptions)
    
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(embeddings)
    
    # 对每个commit计算平均相似度分数（不包括与自身的相似度）
    scores = []
    num_commits = len(commits)
    
    for i in range(num_commits):
        # 移除与自身的相似度（值为1）
        similarities = np.delete(similarity_matrix[i], i)
        # 计算平均相似度
        avg_similarity = np.mean(similarities) if len(similarities) > 0 else 0
        scores.append(avg_similarity)
    
    return scores

def process_task(task_data):
    """处理单个聚类的任务函数"""
    cluster_id, commits = task_data
    
    # 计算每个commit的相似度分数
    similarity_scores = calculate_similarity_scores(commits, MODEL_PATH)
    
    # 为每个commit添加相似度分数字段
    for i, commit in enumerate(commits):
        commit["authority_score"] = float(similarity_scores[i])
    
    # 根据相似度分数降序排序
    sorted_commits = sorted(commits, key=sort_by_authority, reverse=True)
    
    return (cluster_id, sorted_commits)

def main():
    # 确保在主进程中也禁用CUDA
    torch.cuda.is_available = lambda: False
    
    print("Loading data...")
    data = load_json(INPUT_FILE)
    
    print(f"Loading sentence transformer model from {MODEL_PATH} for verification...")
    # 仅用于验证模型可以加载，明确指定使用CPU
    _ = SentenceTransformer(MODEL_PATH, device="cpu")
    
    # 创建结果数据的副本
    result_data = copy.deepcopy(data)
    
    # 准备任务列表
    tasks = []
    for cluster_id, cluster_info in data["clusters"].items():
        tasks.append((cluster_id, cluster_info["commits"]))
    
    print(f"Processing {len(tasks)} clusters with {NUM_WORKERS} workers...")
    
    # 由于我们已经设置了启动方法，这里可以安全使用ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        
        # 使用tqdm显示进度
        for future in tqdm.tqdm(futures, desc="Processing clusters"):
            try:
                cluster_id, sorted_commits = future.result()
                results.append((cluster_id, sorted_commits))
            except Exception as exc:
                print(f"Task generated an exception: {exc}")
                import traceback
                traceback.print_exc()
    
    # 更新结果数据
    for cluster_id, sorted_commits in results:
        result_data["clusters"][cluster_id]["commits"] = sorted_commits
    
    # 保存结果
    print(f"Saving results to {OUTPUT_FILE}...")
    save_json(result_data, OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    main()