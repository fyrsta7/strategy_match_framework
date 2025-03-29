import os
# 在导入任何使用OpenBLAS的库(如numpy)之前设置
os.environ["OPENBLAS_NUM_THREADS"] = "64"  # 设置为64或更小的值

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import umap
from tqdm import tqdm
import pickle
import time
import concurrent.futures
from typing import List, Dict, Any, Tuple
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ==============================================
# 配置参数
# ==============================================
# 用于聚类的 commit 数量
COMMIT_NUM = 40000
# 聚类数量(必须大于1)
CLUSTER_COUNT = 14000
# 使用的 llm 模型，只用于确定文件名
LLM_MODEL = "dsv3"
# 输入输出文件所在的文件夹名
INPUT_DIR_NAME = "partial_commit_40000"
OUTPUT_DIR_NAME = f"kmeans_{COMMIT_NUM}_{LLM_MODEL}_true_{CLUSTER_COUNT}"
# 输入文件路径
INPUT_FILE = config.root_path + f"python/2-general_strategy/result/{INPUT_DIR_NAME}/partial_commit_{COMMIT_NUM}_{LLM_MODEL}_true.json"
# 输出文件路径
OUTPUT_FILE = config.root_path + f"python/2-general_strategy/result/{OUTPUT_DIR_NAME}/init.json"
# 用于存储策略总结的字段名称
OPTIMIZATION_SUMMARY_KEY = "optimization_summary_final"
# 用于存储是否为通用优化策略的字段名称
IS_GENERIC_KEY = "is_generic_optimization_final"
# 使用的预训练模型
MODEL_NAME = "all-MiniLM-L6-v2"
# 缓存文件名
CACHE_FILE = "embeddings_cache.pkl"
# 并行线程数
NUM_THREADS = 32

# 设置随机种子，确保结果可重现
np.random.seed(42)

# 用于JSON序列化的辅助函数
def numpy_json_encoder(obj):
    """处理NumPy类型的JSON编码器"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def load_commits(json_file: str) -> List[Dict[str, Any]]:
    """加载JSON文件中的commits数据"""
    print(f"读取文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        commits = json.load(f)
    return commits

def filter_commits(commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤掉没有优化策略描述的commits"""
    filtered_commits = []
    for commit in commits:
        if OPTIMIZATION_SUMMARY_KEY in commit and commit[OPTIMIZATION_SUMMARY_KEY]:
            filtered_commits.append(commit)
    
    print(f"原始commits数量: {len(commits)}")
    print(f"有优化策略描述的commits数量: {len(filtered_commits)}")
    return filtered_commits

def get_embeddings(commits: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """
    使用预训练的语言模型将优化策略描述转换为向量嵌入
    """
    # 检查是否有缓存的嵌入向量
    if os.path.exists(CACHE_FILE):
        print(f"从缓存文件加载嵌入向量: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
            
            # 验证缓存数据是否与当前数据匹配
            cache_descriptions = cache_data.get('descriptions', [])
            current_descriptions = [commit[OPTIMIZATION_SUMMARY_KEY] for commit in commits]
            
            if len(cache_descriptions) == len(current_descriptions) and all(a == b for a, b in zip(cache_descriptions, current_descriptions)):
                return cache_data['embeddings'], cache_descriptions
            else:
                print("缓存数据与当前数据不匹配，重新计算嵌入向量")
    
    # 使用本地模型路径
    local_model_path = os.path.join(config.root_path, 'models', MODEL_NAME)
    print(f"使用本地模型 '{local_model_path}' 计算嵌入向量...")
    
    model = SentenceTransformer(local_model_path)
    
    # 提取优化策略描述 - 现在是直接字符串而不是列表
    descriptions = [commit[OPTIMIZATION_SUMMARY_KEY] for commit in commits]
    
    # 使用tqdm显示进度
    print("计算嵌入向量中...")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    
    # 缓存计算结果
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'descriptions': descriptions}, f)
    print(f"已将嵌入向量缓存到: {CACHE_FILE}")
    
    return embeddings, descriptions

def parallel_kmeans(k: int, embeddings: np.ndarray) -> Tuple[float, float]:
    """在一个k值上并行运行KMeans聚类并返回结果指标"""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    inertia = kmeans.inertia_
    
    # 计算轮廓系数
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(embeddings, labels)
    
    return inertia, silhouette_avg

def perform_kmeans_clustering(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """执行K均值聚类"""
    print(f"使用 k={n_clusters} 执行K均值聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)

def generate_cluster_results(commits: List[Dict[str, Any]], labels: np.ndarray) -> Dict:
    """生成聚类结果的详细JSON数据"""
    print("生成聚类结果JSON...")
    
    # 创建聚类索引映射
    clusters = {}
    for i, label in enumerate(labels):
        label_str = str(int(label) + 1)  # 将标签转换为从1开始的字符串
        if label_str not in clusters:
            clusters[label_str] = []
        clusters[label_str].append(i)
    
    # 计算每个聚类的大小和generic属性统计
    clusters_by_size = []
    for cluster_id, indices in clusters.items():
        # 获取该聚类中的所有commit
        cluster_commits = [commits[i] for i in indices]
        
        # 统计is_generic_optimization_final属性 - 现在是直接布尔值而不是列表
        generic_true = sum(1 for commit in cluster_commits if commit.get(IS_GENERIC_KEY) is True)
        generic_false = len(cluster_commits) - generic_true
        
        # 添加到排序列表
        clusters_by_size.append({
            "cluster_id": int(cluster_id),
            "commit_num": len(cluster_commits),
            "is_generic_true": generic_true,
            "is_generic_false": generic_false
        })
    
    # 按照commit数量从大到小排序
    clusters_by_size.sort(key=lambda x: x["commit_num"], reverse=True)
    
    # 构建结果对象
    result = {
        'algorithm': 'kmeans',
        'num_clusters': len(clusters),
        'total_commits': len(commits),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'clusters_by_size': clusters_by_size,  # 添加按大小排序的聚类列表
        'clusters': {}
    }
    
    # 填充每个聚类的详情
    for cluster_id, indices in tqdm(clusters.items(), desc="处理聚类"):
        # 收集该聚类的所有commit
        cluster_commits = [commits[i] for i in indices]
        
        # 收集该聚类的所有描述 - 现在是直接字符串而不是列表
        cluster_descriptions = [commit[OPTIMIZATION_SUMMARY_KEY] for commit in cluster_commits]
        
        # 计算该聚类中通用优化策略的比例 - 现在是直接布尔值而不是列表
        generic_count = sum(1 for commit in cluster_commits if commit.get(IS_GENERIC_KEY) is True)
        generic_ratio = float(generic_count / len(cluster_commits) if cluster_commits else 0)
        
        # 统计仓库分布
        repositories = {}
        for commit in cluster_commits:
            repo = commit.get('repository_name', 'unknown')
            if repo not in repositories:
                repositories[repo] = 0
            repositories[repo] += 1
        
        # 将聚类信息添加到结果中
        result['clusters'][cluster_id] = {
            'size': len(cluster_commits),
            'generic_ratio': generic_ratio,
            'repositories': repositories,
            'commits': cluster_commits  # 包含所有原始commit数据
        }
    
    return result

def main():
    """主函数"""
    start_time = time.time()
    print(f"开始K-means聚类分析，使用文件: {INPUT_FILE}")
    
    # 加载并过滤commits
    commits = load_commits(INPUT_FILE)
    filtered_commits = filter_commits(commits)
    
    if not filtered_commits:
        print("没有找到包含优化策略描述的commits，退出")
        return
    
    # 获取嵌入向量
    embeddings, descriptions = get_embeddings(filtered_commits)
    print(f"使用手动指定的聚类数量 = {CLUSTER_COUNT}")
    
    # 执行聚类
    labels = perform_kmeans_clustering(embeddings, CLUSTER_COUNT)
    
    # 生成并保存聚类结果
    result = generate_cluster_results(filtered_commits, labels)
    
    # 保存完整的聚类结果，在文件名中包含commit数量以及聚类数量
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, default=numpy_json_encoder, ensure_ascii=False, indent=2)
    
    print(f"已将完整聚类结果保存为 '{OUTPUT_FILE}'")
    
    # 运行时间统计
    elapsed_time = time.time() - start_time
    print(f"K-means聚类分析完成，总耗时: {elapsed_time:.2f}秒")

if __name__ == "__main__":
    main()