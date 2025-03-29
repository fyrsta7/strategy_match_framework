import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import plotly.express as px
import umap
import hdbscan
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
# 输入文件路径
INPUT_FILE = config.root_path + "python/2-general_strategy/partial_commit_1000.json"
# 密度聚类的最小样本数
MIN_SAMPLES = 5
# 密度聚类的最小聚类大小
MIN_CLUSTER_SIZE = 5
# 使用的预训练模型
MODEL_NAME = "all-MiniLM-L6-v2"
# 缓存文件名
CACHE_FILE = "embeddings_cache.pkl"
# 输出文件前缀
OUTPUT_PREFIX = "density_cluster"
# UMAP降维设置
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.0
UMAP_N_COMPONENTS = 10
# 并行线程数
NUM_THREADS = 128

# 设置随机种子，确保结果可重现
np.random.seed(42)

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
        if 'optimization_summary' in commit and commit['optimization_summary']:
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
            current_descriptions = [commit['optimization_summary'] for commit in commits]
            
            if len(cache_descriptions) == len(current_descriptions) and all(a == b for a, b in zip(cache_descriptions, current_descriptions)):
                return cache_data['embeddings'], cache_descriptions
            else:
                print("缓存数据与当前数据不匹配，重新计算嵌入向量")
    
    # 使用本地模型路径
    local_model_path = os.path.join(config.root_path, 'models', MODEL_NAME)
    print(f"使用本地模型 '{local_model_path}' 计算嵌入向量...")
    
    model = SentenceTransformer(local_model_path)
    
    # 提取优化策略描述
    descriptions = [commit['optimization_summary'] for commit in commits]
    
    # 使用tqdm显示进度
    print("计算嵌入向量中...")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    
    # 缓存计算结果
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'descriptions': descriptions}, f)
    print(f"已将嵌入向量缓存到: {CACHE_FILE}")
    
    return embeddings, descriptions

def reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    """使用UMAP进行降维，为密度聚类做准备"""
    print(f"使用UMAP将{embeddings.shape[1]}维向量降至{UMAP_N_COMPONENTS}维...")
    
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        random_state=42
    )
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"降维完成: {reduced_embeddings.shape}")
    
    return reduced_embeddings

def perform_density_clustering(embeddings: np.ndarray) -> np.ndarray:
    """执行HDBSCAN密度聚类"""
    print(f"执行HDBSCAN密度聚类 (min_samples={MIN_SAMPLES}, min_cluster_size={MIN_CLUSTER_SIZE})...")
    
    # 应用HDBSCAN进行聚类
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        gen_min_span_tree=True,
        prediction_data=True
    )
    
    labels = clusterer.fit_predict(embeddings)
    
    # 统计聚类结果
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"HDBSCAN聚类结果:")
    print(f"- 聚类数量: {n_clusters}")
    print(f"- 未分类样本数量: {n_noise} ({n_noise/len(labels):.1%})")
    
    return labels

def visualize_clusters(original_embeddings: np.ndarray, labels: np.ndarray, 
                      descriptions: List[str], commits: List[Dict[str, Any]]) -> None:
    """使用UMAP和Plotly可视化聚类结果"""
    print("使用UMAP进行2D可视化...")
    umap_2d = umap.UMAP(n_neighbors=30, min_dist=0.1, 
                       n_components=2, random_state=42).fit_transform(original_embeddings)
    
    # 创建包含所需数据的DataFrame
    df = pd.DataFrame({
        'x': umap_2d[:, 0],
        'y': umap_2d[:, 1],
        'cluster': labels,
        'description': descriptions,
        'repository': [commit.get('repository_name', 'unknown') for commit in commits],
        'commit_hash': [commit.get('hash', 'unknown') for commit in commits],
        'is_generic': [commit.get('is_generic_optimization', False) for commit in commits]
    })
    
    # 为可能的噪声点和聚类添加标签
    df['cluster_label'] = df['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x+1}')
    
    # 创建散点图
    fig = px.scatter(
        df, x='x', y='y', color='cluster_label', hover_data=['description', 'repository', 'commit_hash'],
        symbol='is_generic', symbol_map={True: 'circle', False: 'x'},
        title='密度聚类优化策略结果'
    )
    
    # 保存交互式HTML文件
    html_file = f'{OUTPUT_PREFIX}_visualization.html'
    fig.write_html(html_file)
    print(f"已将交互式可视化结果保存为 '{html_file}'")
    
    # 保存为静态图像
    png_file = f'{OUTPUT_PREFIX}_visualization.png'
    fig.write_image(png_file)
    print(f"已将静态可视化结果保存为 '{png_file}'")
    
    # 计算每个聚类的规模
    cluster_sizes = df['cluster_label'].value_counts().reset_index()
    cluster_sizes.columns = ['Cluster', 'Count']
    
    # 创建柱状图
    fig = px.bar(
        cluster_sizes, x='Cluster', y='Count',
        title='密度聚类规模分布'
    )
    
    # 保存柱状图
    fig.write_html(f'{OUTPUT_PREFIX}_sizes.html')
    fig.write_image(f'{OUTPUT_PREFIX}_sizes.png')

def generate_cluster_results(commits: List[Dict[str, Any]], labels: np.ndarray) -> Dict:
    """生成聚类结果的详细JSON数据"""
    print("生成聚类结果JSON...")
    
    # 创建聚类索引映射
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            label_str = "noise"  # 噪声点
        else:
            label_str = str(int(label) + 1)  # 将标签转换为从1开始的字符串
        
        if label_str not in clusters:
            clusters[label_str] = []
        clusters[label_str].append(i)
    
    # 构建结果对象
    result = {
        'algorithm': 'hdbscan',
        'min_samples': MIN_SAMPLES,
        'min_cluster_size': MIN_CLUSTER_SIZE,
        'num_clusters': len(clusters) - (1 if "noise" in clusters else 0),
        'total_commits': len(commits),
        'noise_points': len(clusters.get("noise", [])),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'clusters': {}
    }
    
    # 填充每个聚类的详情
    for cluster_id, indices in tqdm(clusters.items(), desc="处理聚类"):
        # 收集该聚类的所有commit
        cluster_commits = [commits[i] for i in indices]
        
        # 收集该聚类的所有描述
        cluster_descriptions = [commit['optimization_summary'] for commit in cluster_commits]
        
        # 计算该聚类中通用优化策略的比例
        generic_count = sum(1 for commit in cluster_commits if commit.get('is_generic_optimization', False))
        generic_ratio = generic_count / len(cluster_commits) if cluster_commits else 0
        
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

def analyze_cluster_params():
    """分析不同参数组合对聚类结果的影响"""
    # 此功能可在未来扩展，目前仅用默认参数
    pass

def main():
    """主函数"""
    start_time = time.time()
    print(f"开始密度聚类分析，使用文件: {INPUT_FILE}")
    
    # 加载并过滤commits
    commits = load_commits(INPUT_FILE)
    filtered_commits = filter_commits(commits)
    
    if not filtered_commits:
        print("没有找到包含优化策略描述的commits，退出")
        return
    
    # 获取嵌入向量
    original_embeddings, descriptions = get_embeddings(filtered_commits)
    
    # 使用UMAP降维
    reduced_embeddings = reduce_dimensions(original_embeddings)
    
    # 执行密度聚类
    labels = perform_density_clustering(reduced_embeddings)
    
    # 可视化聚类结果
    visualize_clusters(original_embeddings, labels, descriptions, filtered_commits)
    
    # 生成并保存聚类结果
    result = generate_cluster_results(filtered_commits, labels)
    
    # 保存完整的聚类结果
    with open(f'{OUTPUT_PREFIX}_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"已将完整聚类结果保存为 '{OUTPUT_PREFIX}_results.json'")
    
    # 运行时间统计
    elapsed_time = time.time() - start_time
    print(f"密度聚类分析完成，总耗时: {elapsed_time:.2f}秒")

if __name__ == "__main__":
    main()