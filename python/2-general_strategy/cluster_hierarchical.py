import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import plotly.express as px
import umap
from tqdm import tqdm
import pickle
import time
import concurrent.futures
from typing import List, Dict, Any, Tuple
from scipy.cluster.hierarchy import dendrogram
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ==============================================
# 配置参数
# ==============================================
# 输入文件路径
INPUT_FILE = config.root_path + "python/2-general_strategy/partial_commit_1000.json"
# 指定聚类数量(0表示自动确定)
NUM_CLUSTERS = 0
# 用于自动确定k值时的最大尝试数
MAX_K = 15
# 使用的预训练模型
MODEL_NAME = "all-MiniLM-L6-v2"
# 缓存文件名
CACHE_FILE = "embeddings_cache.pkl"
# 输出文件前缀
OUTPUT_PREFIX = "hierarchical_cluster"
# 并行线程数
NUM_THREADS = 128
# 层次聚类联接方法 ('ward', 'complete', 'average', 'single')
LINKAGE = 'ward'

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

def parallel_hierarchical(k: int, embeddings: np.ndarray) -> float:
    """在一个k值上并行运行层次聚类并返回轮廓系数"""
    model = AgglomerativeClustering(n_clusters=k, linkage=LINKAGE)
    labels = model.fit_predict(embeddings)
    silhouette_avg = silhouette_score(embeddings, labels)
    return silhouette_avg

def find_optimal_k(embeddings: np.ndarray) -> int:
    """
    使用轮廓系数找到最佳的聚类数量k
    """
    print("寻找层次聚类的最佳聚类数量...")
    
    # 计算不同k值的层次聚类结果
    silhouette_scores = []
    k_values = range(2, MAX_K + 1)
    
    # 使用线程池并行计算
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(parallel_hierarchical, k, embeddings): k for k in k_values}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="评估k值"):
            k = futures[future]
            try:
                silhouette = future.result()
                silhouette_scores.append((k, silhouette))
            except Exception as e:
                print(f"计算k={k}时发生错误: {str(e)}")
                silhouette_scores.append((k, -1))
    
    # 将结果按k值排序
    silhouette_scores.sort(key=lambda x: x[0])
    k_values = [k for k, _ in silhouette_scores]
    s_scores = [s for _, s in silhouette_scores]
    
    # 创建轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, s_scores, 'r-o')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('轮廓系数')
    plt.title('层次聚类的轮廓系数分析')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}_optimal_k.png')
    print(f"已将最佳k值分析图保存为 '{OUTPUT_PREFIX}_optimal_k.png'")
    
    # 根据轮廓系数选择最佳k值
    best_k_index = np.argmax(s_scores)
    best_k = k_values[best_k_index]
    print(f"推荐的最佳聚类数量 k = {best_k}，轮廓系数 = {s_scores[best_k_index]:.3f}")
    
    # 将k值分析结果保存为JSON
    k_analysis = {
        'k_values': k_values,
        'silhouette_scores': s_scores,
        'best_k': best_k,
        'best_silhouette': s_scores[best_k_index]
    }
    
    with open(f'{OUTPUT_PREFIX}_k_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(k_analysis, f, indent=2)
    
    return best_k

def perform_hierarchical_clustering(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """执行层次聚类"""
    print(f"使用 k={n_clusters} 执行层次聚类 (linkage={LINKAGE})...")
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=LINKAGE)
    return model.fit_predict(embeddings)

def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, 
                      descriptions: List[str], commits: List[Dict[str, Any]]) -> None:
    """使用UMAP和Plotly可视化聚类结果"""
    print("使用UMAP降维以便可视化...")
    umap_embeddings = umap.UMAP(n_neighbors=30, min_dist=0.1, 
                               n_components=2, random_state=42).fit_transform(embeddings)
    
    # 创建包含所需数据的DataFrame
    df = pd.DataFrame({
        'x': umap_embeddings[:, 0],
        'y': umap_embeddings[:, 1],
        'cluster': labels,
        'description': descriptions,
        'repository': [commit.get('repository_name', 'unknown') for commit in commits],
        'commit_hash': [commit.get('hash', 'unknown') for commit in commits],
        'is_generic': [commit.get('is_generic_optimization', False) for commit in commits]
    })
    
    # 将cluster标签格式化为从1开始的序号
    df['cluster_label'] = df['cluster'].apply(lambda x: f'Cluster {x+1}')
    
    # 创建散点图
    fig = px.scatter(
        df, x='x', y='y', color='cluster_label', hover_data=['description', 'repository', 'commit_hash'],
        symbol='is_generic', symbol_map={True: 'circle', False: 'x'},
        title='层次聚类优化策略结果'
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
        title='层次聚类规模分布'
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
        label_str = str(int(label) + 1)  # 将标签转换为从1开始的字符串
        if label_str not in clusters:
            clusters[label_str] = []
        clusters[label_str].append(i)
    
    # 构建结果对象
    result = {
        'algorithm': 'hierarchical',
        'linkage': LINKAGE,
        'num_clusters': len(clusters),
        'total_commits': len(commits),
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

def main():
    """主函数"""
    start_time = time.time()
    print(f"开始层次聚类分析，使用文件: {INPUT_FILE}")
    
    # 加载并过滤commits
    commits = load_commits(INPUT_FILE)
    filtered_commits = filter_commits(commits)
    
    if not filtered_commits:
        print("没有找到包含优化策略描述的commits，退出")
        return
    
    # 获取嵌入向量
    embeddings, descriptions = get_embeddings(filtered_commits)
    
    # 确定聚类数量
    k = NUM_CLUSTERS if NUM_CLUSTERS > 0 else find_optimal_k(embeddings)
    
    # 执行聚类
    labels = perform_hierarchical_clustering(embeddings, k)
    
    # 可视化聚类结果
    visualize_clusters(embeddings, labels, descriptions, filtered_commits)
    
    # 生成并保存聚类结果
    result = generate_cluster_results(filtered_commits, labels)
    
    # 保存完整的聚类结果
    with open(f'{OUTPUT_PREFIX}_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"已将完整聚类结果保存为 '{OUTPUT_PREFIX}_results.json'")
    
    # 运行时间统计
    elapsed_time = time.time() - start_time
    print(f"层次聚类分析完成，总耗时: {elapsed_time:.2f}秒")

if __name__ == "__main__":
    main()