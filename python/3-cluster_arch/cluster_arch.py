import os
import json
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import pickle
from pathlib import Path
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 聚类控制参数 ============
MIN_SIMILARITY_THRESHOLD = 0.89  # 主控参数：聚类内最小相似度 (0-1)
MIN_CLUSTER_SIZE = 2            # 形成聚类的最小commit数
MAX_CLUSTERS = None             # 最大聚类数限制（None为不限制）

# ============ 筛选控制参数 ============
FILTER_COMMITS_BY_FILES = False  # 是否启用commit文件筛选功能

# ============ 算法选择 ============
CLUSTERING_ALGORITHM = "DBSCAN"  # 可选: "DBSCAN", "AgglomerativeClustering"

# ============ 输出控制 ============
EXPORT_DETAILED_RESULTS = True  # 导出的聚类结果中是否包含属于噪点的commit

# ============ 缓存配置 ============
EMBEDDING_CACHE_DIR = os.path.join(config.root_path, "cache", "embeddings_arch")
CACHE_VERSION = "v1.0"
BATCH_SIZE = 64

# ============ 输出配置 ============
OUTPUT_DIR = os.path.join(config.root_path, "python", "3-cluster_arch", "result_30342_arch")
KNOWLEDGE_BASE_ROOT = os.path.join(config.root_path, "knowledge_base")
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
JSON_FILE_NAME = "summary_filter_arch.json"

# ============ 模型配置 ============
model_path = os.path.join(config.root_path, "models/all-MiniLM-L6-v2")
sentence_model = SentenceTransformer(model_path)

# 线程锁用于缓存安全
cache_lock = threading.Lock()

def generate_cache_key(text, model_name="all-MiniLM-L6-v2"):
    """生成缓存键：基于文本内容+模型名+版本"""
    content = f"{CACHE_VERSION}|{model_name}|{text}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_cache_path(cache_key):
    """获取缓存文件路径，按子目录分层存储"""
    subdir = cache_key[:2]  # 前两位作为子目录，避免单目录文件过多
    return os.path.join(EMBEDDING_CACHE_DIR, subdir, f"{cache_key}.pkl")

class EmbeddingCache:
    def __init__(self, cache_dir, model_name):
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.memory_cache = {}  # 内存二级缓存
        self.hit_count = 0
        self.miss_count = 0
        
        # 确保缓存目录存在
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def get_embedding(self, text):
        """获取embedding，优先从缓存读取"""
        cache_key = generate_cache_key(text, self.model_name)
        
        with cache_lock:
            # 1. 内存缓存查找
            if cache_key in self.memory_cache:
                self.hit_count += 1
                return self.memory_cache[cache_key]
        
        # 2. 磁盘缓存查找
        cache_path = get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                with cache_lock:
                    self.memory_cache[cache_key] = embedding  # 加载到内存缓存
                    self.hit_count += 1
                return embedding
            except Exception as e:
                print(f"缓存读取失败: {e}")
        
        # 3. 缓存未命中
        with cache_lock:
            self.miss_count += 1
        return None
    
    def save_embedding(self, text, embedding):
        """保存embedding到缓存"""
        cache_key = generate_cache_key(text, self.model_name)
        cache_path = get_cache_path(cache_key)
        
        # 确保子目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # 保存到磁盘
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"缓存保存失败: {e}")
        
        # 保存到内存缓存
        with cache_lock:
            self.memory_cache[cache_key] = embedding
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        with cache_lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            return {
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "memory_cache_size": len(self.memory_cache)
            }

# 初始化缓存管理器
embedding_cache = EmbeddingCache(EMBEDDING_CACHE_DIR, "all-MiniLM-L6-v2")

def get_embedding_with_cache(text):
    """带缓存的embedding获取函数"""
    # 先尝试从缓存获取
    cached_embedding = embedding_cache.get_embedding(text)
    if cached_embedding is not None:
        return cached_embedding
    
    # 缓存未命中，计算新embedding
    embedding = sentence_model.encode([text])[0]
    
    # 保存到缓存
    embedding_cache.save_embedding(text, embedding)
    
    return embedding

def batch_get_embeddings(texts, batch_size=BATCH_SIZE):
    """批量获取embeddings，优化缓存命中率"""
    embeddings = []
    uncached_texts = []
    uncached_indices = []
    
    # 第一轮：检查缓存
    for i, text in enumerate(texts):
        cached_embedding = embedding_cache.get_embedding(text)
        if cached_embedding is not None:
            embeddings.append(cached_embedding)
        else:
            embeddings.append(None)  # 占位
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # 第二轮：批量计算未缓存的embeddings
    if uncached_texts:
        print(f"计算 {len(uncached_texts)} 个新embeddings...")
        new_embeddings = sentence_model.encode(uncached_texts, batch_size=batch_size)
        
        # 保存新计算的embeddings并填充结果
        for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
            embedding_cache.save_embedding(text, embedding)
            embeddings[uncached_indices[idx]] = embedding
    
    return embeddings

def is_valid_commit_with_files(commit, repo_path):
    """验证commit是否包含必要的字段和文件，返回(是否有效, 失败原因)"""
    try:
        # 1. 检查必要字段
        if 'func_start_line' not in commit or not commit['func_start_line']:
            return False, "缺少func_start_line字段"
        if 'func_end_line' not in commit or not commit['func_end_line']:
            return False, "缺少func_end_line字段"
        
        # 2. 检查必要文件
        hash = commit.get('hash', '')
        if not hash:
            return False, "缺少hash字段"
        
        commit_dir = os.path.join(repo_path, "modified_file", hash)
        if not os.path.exists(commit_dir):
            return False, "commit目录不存在"
        
        # 检查diff.txt文件
        diff_file = os.path.join(commit_dir, 'diff.txt')
        if not os.path.exists(diff_file):
            return False, "缺少diff.txt文件"
        
        # 检查before_func文件（文件名为before_func，但后缀不确定）
        before_func_found = False
        for filename in os.listdir(commit_dir):
            if filename.startswith('before_func'):
                before_func_found = True
                break
        
        if not before_func_found:
            return False, "缺少before_func文件"
        
        return True, ""
        
    except Exception as e:
        # 如果出现任何异常，认为该commit无效
        return False, f"异常错误: {str(e)}"

def load_all_commits():
    """加载所有代码库的commit数据"""
    print(f"从知识库加载commit数据: {KNOWLEDGE_BASE_ROOT}")
    print(f"使用代码库列表: {REPO_LIST_FILE}")
    
    # 读取代码库列表
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件不存在 - {REPO_LIST_FILE}")
        return []
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repo_list = json.load(f)
    
    # 获取代码库名称列表
    repositories = [repo.get('name_long', repo.get('name', '')) for repo in repo_list if repo.get('name_long') or repo.get('name')]
    
    all_commits = []
    stats = {
        "total_repos": len(repositories), 
        "valid_repos": 0, 
        "total_commits": 0,
        "summary_filtered": 0,
        "file_filtered": 0 if FILTER_COMMITS_BY_FILES else None
    }
    
    # 筛选失败原因统计
    filter_failure_stats = {}
    
    # 创建进度条
    repo_pbar = tqdm(repositories, desc="加载代码库", unit="repo")
    
    for repo_name in repo_pbar:
        repo_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name)
        json_file_path = os.path.join(repo_path, JSON_FILE_NAME)
        
        # 检查文件是否存在
        if not os.path.exists(json_file_path):
            continue
        
        try:
            # 读取commit数据
            with open(json_file_path, 'r', encoding='utf-8') as f:
                commits = json.load(f)
            
            # 第一层过滤：检查optimization_summary_arch_final字段
            summary_valid_commits = []
            for commit in commits:
                if 'optimization_summary_arch_final' in commit and commit['optimization_summary_arch_final'].strip():
                    summary_valid_commits.append(commit)
            
            stats["summary_filtered"] += len(summary_valid_commits)
            
            # 第二层过滤：如果启用文件筛选，检查字段和文件
            final_valid_commits = []
            if FILTER_COMMITS_BY_FILES:
                for commit in summary_valid_commits:
                    is_valid, failure_reason = is_valid_commit_with_files(commit, repo_path)
                    if is_valid:
                        final_valid_commits.append(commit)
                    else:
                        # 统计失败原因
                        if failure_reason in filter_failure_stats:
                            filter_failure_stats[failure_reason] += 1
                        else:
                            filter_failure_stats[failure_reason] = 1
                stats["file_filtered"] += len(final_valid_commits)
            else:
                final_valid_commits = summary_valid_commits
            
            if final_valid_commits:
                all_commits.extend(final_valid_commits)
                stats["valid_repos"] += 1
                stats["total_commits"] += len(final_valid_commits)
                
                repo_pbar.set_postfix({
                    "有效commits": len(final_valid_commits),
                    "总计": stats["total_commits"]
                })
        
        except Exception as e:
            print(f"\n[错误] 读取 {repo_name} 失败: {str(e)}")
            continue
    
    repo_pbar.close()
    
    print(f"\n=== 数据加载统计 ===")
    print(f"总代码库数: {stats['total_repos']}")
    print(f"有效代码库数: {stats['valid_repos']}")
    print(f"总结字段有效: {stats['summary_filtered']}")
    if FILTER_COMMITS_BY_FILES:
        print(f"文件筛选后: {stats['file_filtered']}")
        print(f"文件筛选率: {stats['file_filtered']/stats['summary_filtered']:.2%}" if stats['summary_filtered'] > 0 else "N/A")
        
        # 输出筛选失败原因统计
        if filter_failure_stats:
            print(f"\n=== 文件筛选失败原因统计 ===")
            total_failed = sum(filter_failure_stats.values())
            print(f"总筛选失败数: {total_failed}")
            # 按失败数量排序
            sorted_failures = sorted(filter_failure_stats.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_failures:
                percentage = count / total_failed * 100
                print(f"  {reason}: {count} ({percentage:.1f}%)")
    
    print(f"最终commit数: {stats['total_commits']}")
    
    return all_commits

def perform_clustering(embeddings, min_similarity_threshold):
    """执行聚类算法"""
    print(f"\n开始聚类分析...")
    print(f"相似度阈值: {min_similarity_threshold}")
    print(f"聚类算法: {CLUSTERING_ALGORITHM}")
    
    if CLUSTERING_ALGORITHM == "DBSCAN":
        # DBSCAN参数转换
        eps = 1 - min_similarity_threshold
        clustering = DBSCAN(
            eps=eps, 
            min_samples=MIN_CLUSTER_SIZE, 
            metric='cosine'
        )
        
        print(f"DBSCAN参数: eps={eps:.3f}, min_samples={MIN_CLUSTER_SIZE}")
        
        # 执行聚类
        cluster_labels = clustering.fit_predict(embeddings)
        
    else:
        raise ValueError(f"不支持的聚类算法: {CLUSTERING_ALGORITHM}")
    
    return cluster_labels

def analyze_clustering_results(cluster_labels, commits):
    """分析聚类结果"""
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\n=== 聚类结果统计 ===")
    print(f"聚类数量: {n_clusters}")
    print(f"噪声点数量: {n_noise}")
    print(f"聚类占比: {(len(commits) - n_noise) / len(commits):.2%}")
    
    # 统计每个聚类的大小
    cluster_sizes = {}
    for label in cluster_labels:
        if label != -1:  # 排除噪声点
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
    
    if cluster_sizes:
        print(f"平均聚类大小: {np.mean(list(cluster_sizes.values())):.1f}")
        print(f"最大聚类大小: {max(cluster_sizes.values())}")
        print(f"最小聚类大小: {min(cluster_sizes.values())}")
    
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sizes": cluster_sizes
    }

def organize_clustering_results(cluster_labels, commits, embeddings):
    """组织聚类结果为输出格式"""
    print(f"\n组织聚类结果...")
    
    # 按聚类标签分组
    clusters_dict = {}
    noise_points = []
    
    for i, (label, commit) in enumerate(zip(cluster_labels, commits)):
        if label == -1:  # 噪声点
            noise_points.append(commit)
        else:
            if label not in clusters_dict:
                clusters_dict[label] = {
                    "commits": [],
                    "embeddings": []
                }
            clusters_dict[label]["commits"].append(commit)
            clusters_dict[label]["embeddings"].append(embeddings[i])
    
    # 转换为最终格式
    clusters = []
    cluster_pbar = tqdm(clusters_dict.items(), desc="处理聚类", unit="cluster")
    
    for cluster_id, cluster_data in cluster_pbar:
        cluster_commits = cluster_data["commits"]
        cluster_embeddings = cluster_data["embeddings"]
        
        cluster = {
            "cluster_id": int(cluster_id),
            "size": len(cluster_commits),
            "commits": cluster_commits
        }
        clusters.append(cluster)
        
        cluster_pbar.set_postfix({"大小": len(cluster_commits)})
    
    cluster_pbar.close()
    
    # 按聚类大小排序
    clusters.sort(key=lambda x: x["size"], reverse=True)
    
    # 重新编号：按顺序从1开始递增
    for index, cluster in enumerate(clusters):
        cluster["cluster_id"] = index + 1
    
    return clusters, noise_points

def save_results(clusters, noise_points, stats, min_similarity_threshold):
    """保存聚类结果"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 将小数转换为字符串，替换小数点为下划线，去掉末尾的0
    threshold_str = f"{min_similarity_threshold:.10f}".rstrip('0').rstrip('.')
    # 生成输出文件名
    output_filename = f"{threshold_str.replace('.', '_')}_{MIN_CLUSTER_SIZE}_{FILTER_COMMITS_BY_FILES}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 组织输出数据
    output_data = {
        "clustering_config": {
            "min_similarity_threshold": min_similarity_threshold,
            "algorithm": CLUSTERING_ALGORITHM,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "filter_commits_by_files": FILTER_COMMITS_BY_FILES,
            "total_commits": stats["n_clusters"] * 0 + len(noise_points) + sum(cluster["size"] for cluster in clusters),
            "num_clusters": stats["n_clusters"],
            "noise_points": stats["n_noise"],
            "clustering_rate": (sum(cluster["size"] for cluster in clusters)) / (sum(cluster["size"] for cluster in clusters) + len(noise_points)) if (sum(cluster["size"] for cluster in clusters) + len(noise_points)) > 0 else 0
        },
        "clusters": clusters,
        "noise_points": noise_points if EXPORT_DETAILED_RESULTS else []
    }
    
    # 保存到文件
    print(f"\n保存结果到: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"结果保存成功！")
        return output_path
    except Exception as e:
        print(f"保存失败: {str(e)}")
        return None

def main():
    """主函数"""
    print(f"=== 体系结构相关策略聚类分析 ===")
    print(f"相似度阈值: {MIN_SIMILARITY_THRESHOLD}")
    print(f"最小聚类大小: {MIN_CLUSTER_SIZE}")
    print(f"文件筛选: {'启用' if FILTER_COMMITS_BY_FILES else '禁用'}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 1. 加载所有commit数据
    all_commits = load_all_commits()
    if not all_commits:
        print("未找到有效的commit数据！")
        return
    
    # 2. 获取embeddings
    print(f"\n计算文本embeddings...")
    texts = [commit['optimization_summary_arch_final'] for commit in all_commits]
    embeddings = batch_get_embeddings(texts)
    
    # 输出缓存统计
    cache_stats = embedding_cache.get_cache_stats()
    print(f"缓存统计: 命中率 {cache_stats['hit_rate']:.2%} "
          f"(命中: {cache_stats['hit_count']}, 未命中: {cache_stats['miss_count']})")
    
    # 3. 执行聚类
    cluster_labels = perform_clustering(embeddings, MIN_SIMILARITY_THRESHOLD)
    
    # 4. 分析结果
    stats = analyze_clustering_results(cluster_labels, all_commits)
    
    # 5. 组织结果
    clusters, noise_points = organize_clustering_results(cluster_labels, all_commits, embeddings)
    
    # 6. 保存结果
    output_path = save_results(clusters, noise_points, stats, MIN_SIMILARITY_THRESHOLD)
    
    if output_path:
        print(f"\n=== 聚类完成 ===")
        print(f"输出文件: {output_path}")
        print(f"聚类数量: {len(clusters)}")
        print(f"最大聚类大小: {max(cluster['size'] for cluster in clusters) if clusters else 0}")

if __name__ == "__main__":
    main()

