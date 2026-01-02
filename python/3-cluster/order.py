import os
import json
import sys
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import pickle
from pathlib import Path
import threading
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置参数 ============
EMBEDDING_CACHE_DIR = os.path.join(config.root_path, "cache", "embeddings")
CACHE_VERSION = "v1.0"
BATCH_SIZE = 32

# ============ 输入输出配置 ============
INPUT_DIR = os.path.join(config.root_path, "python", "2-cluster_new", "result_30342_old")
INPUT_FILE = "0_85_5_False.json"
OUTPUT_DIR = INPUT_DIR  # 输出到同一目录

# ============ 模型配置 ============
model_path = os.path.join(config.root_path, "models/all-MiniLM-L6-v2")
# sentence_model = SentenceTransformer(model_path)
# 强制使用cpu
sentence_model = SentenceTransformer(model_path, device='cpu')

# 线程锁用于缓存安全
cache_lock = threading.Lock()

def generate_cache_key(text, model_name="all-MiniLM-L6-v2"):
    """生成缓存键：基于文本内容+模型名+版本"""
    content = f"{CACHE_VERSION}|{model_name}|{text}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_cache_path(cache_key):
    """获取缓存文件路径，按子目录分层存储"""
    subdir = cache_key[:2]
    return os.path.join(EMBEDDING_CACHE_DIR, subdir, f"{cache_key}.pkl")

class EmbeddingCache:
    def __init__(self, cache_dir, model_name):
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.memory_cache = {}
        self.hit_count = 0
        self.miss_count = 0
        
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def get_embedding(self, text):
        """获取embedding，优先从缓存读取"""
        cache_key = generate_cache_key(text, self.model_name)
        
        with cache_lock:
            if cache_key in self.memory_cache:
                self.hit_count += 1
                return self.memory_cache[cache_key]
        
        cache_path = get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                with cache_lock:
                    self.memory_cache[cache_key] = embedding
                    self.hit_count += 1
                return embedding
            except Exception as e:
                print(f"缓存读取失败: {e}")
        
        with cache_lock:
            self.miss_count += 1
        return None
    
    def save_embedding(self, text, embedding):
        """保存embedding到缓存"""
        cache_key = generate_cache_key(text, self.model_name)
        cache_path = get_cache_path(cache_key)
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"缓存保存失败: {e}")
        
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
    cached_embedding = embedding_cache.get_embedding(text)
    if cached_embedding is not None:
        return cached_embedding
    
    embedding = sentence_model.encode([text])[0]
    embedding_cache.save_embedding(text, embedding)
    
    return embedding

def batch_get_embeddings(texts, batch_size=BATCH_SIZE):
    """批量获取embeddings，优化缓存命中率"""
    embeddings = []
    uncached_texts = []
    uncached_indices = []
    
    # 检查缓存
    for i, text in enumerate(texts):
        cached_embedding = embedding_cache.get_embedding(text)
        if cached_embedding is not None:
            embeddings.append(cached_embedding)
        else:
            embeddings.append(None)
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # 批量计算未缓存的embeddings
    if uncached_texts:
        print(f"  计算 {len(uncached_texts)} 个新embeddings...")
        new_embeddings = sentence_model.encode(uncached_texts, batch_size=batch_size)
        
        for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
            embedding_cache.save_embedding(text, embedding)
            embeddings[uncached_indices[idx]] = embedding
    
    return embeddings

def calculate_centrality_score(target_embedding, cluster_embeddings, target_index):
    """计算commit在聚类中的中心性得分（平均相似度）"""
    similarities = []
    
    for i, other_embedding in enumerate(cluster_embeddings):
        if i != target_index:  # 排除自己
            similarity = cosine_similarity([target_embedding], [other_embedding])[0][0]
            similarities.append(similarity)
    
    if not similarities:
        return 0.0
    
    return np.mean(similarities)

def rank_cluster_commits(cluster_commits):
    """对聚类内的commits按中心性进行排序"""
    if len(cluster_commits) <= 1:
        # 单个commit或空聚类，直接返回
        for commit in cluster_commits:
            commit['representativeness_score'] = 1.0
        return cluster_commits
    
    # 获取所有commit的embeddings
    texts = [commit['optimization_summary_final'] for commit in cluster_commits]
    embeddings = batch_get_embeddings(texts)
    
    # 计算每个commit的中心性得分
    scores = []
    for i, (commit, embedding) in enumerate(zip(cluster_commits, embeddings)):
        centrality_score = calculate_centrality_score(embedding, embeddings, i)
        scores.append((commit, centrality_score))
    
    # 按得分降序排序
    ranked_commits = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # 添加得分信息到commit数据中
    result_commits = []
    for commit, score in ranked_commits:
        commit['representativeness_score'] = float(score)
        result_commits.append(commit)
    
    return result_commits

def load_clustering_results(input_file):
    """加载聚类结果文件"""
    input_path = os.path.join(INPUT_DIR, input_file)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    print(f"加载聚类结果: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"聚类数量: {len(data.get('clusters', []))}")
        print(f"噪声点数量: {len(data.get('noise_points', []))}")
        
        return data
    except Exception as e:
        raise Exception(f"读取聚类结果文件失败: {str(e)}")

def process_clusters(clustering_data):
    """处理所有聚类，对每个聚类内的commits进行排序"""
    clusters = clustering_data.get('clusters', [])
    
    if not clusters:
        print("未找到任何聚类数据")
        return clustering_data
    
    print(f"\n开始处理 {len(clusters)} 个聚类...")
    
    # 统计信息
    total_commits_processed = 0
    clusters_with_ranking = 0
    
    # 创建进度条
    cluster_pbar = tqdm(clusters, desc="处理聚类", unit="cluster")
    
    for cluster in cluster_pbar:
        cluster_id = cluster.get('cluster_id', 'unknown')
        cluster_commits = cluster.get('commits', [])
        
        if len(cluster_commits) <= 1:
            # 单个commit的聚类无需排序
            if cluster_commits:
                cluster_commits[0]['representativeness_score'] = 1.0
            cluster_pbar.set_postfix({"聚类ID": cluster_id, "大小": len(cluster_commits), "状态": "跳过"})
            continue
        
        try:
            # 对聚类内commits进行排序
            ranked_commits = rank_cluster_commits(cluster_commits)
            
            # 更新聚类数据
            cluster['commits'] = ranked_commits
            cluster['most_representative_commit'] = ranked_commits[0] if ranked_commits else None
            cluster['avg_representativeness'] = np.mean([c['representativeness_score'] for c in ranked_commits])
            
            total_commits_processed += len(ranked_commits)
            clusters_with_ranking += 1
            
            cluster_pbar.set_postfix({
                "聚类ID": cluster_id, 
                "大小": len(ranked_commits), 
                "最高分": f"{ranked_commits[0]['representativeness_score']:.3f}" if ranked_commits else "0"
            })
            
        except Exception as e:
            print(f"\n[错误] 处理聚类 {cluster_id} 失败: {str(e)}")
            cluster_pbar.set_postfix({"聚类ID": cluster_id, "状态": "失败"})
            continue
    
    cluster_pbar.close()
    
    print(f"\n=== 排序完成统计 ===")
    print(f"处理的聚类数: {clusters_with_ranking}/{len(clusters)}")
    print(f"排序的commits数: {total_commits_processed}")
    
    return clustering_data

def save_ranked_results(ranked_data, input_filename):
    """保存排序后的结果"""
    # 生成输出文件名（输入文件名 + "_order"）
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_order.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n保存排序结果到: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ranked_data, f, ensure_ascii=False, indent=2)
        
        print(f"排序结果保存成功！")
        return output_path
        
    except Exception as e:
        print(f"保存失败: {str(e)}")
        return None

def analyze_ranking_results(ranked_data):
    """分析排序结果的统计信息"""
    clusters = ranked_data.get('clusters', [])
    
    if not clusters:
        return
    
    print(f"\n=== 排序结果分析 ===")
    
    # 统计信息
    total_clusters = len(clusters)
    total_commits = sum(len(cluster.get('commits', [])) for cluster in clusters)
    ranked_clusters = sum(1 for cluster in clusters if len(cluster.get('commits', [])) > 1)
    
    # 代表性得分统计
    all_scores = []
    cluster_avg_scores = []
    
    for cluster in clusters:
        commits = cluster.get('commits', [])
        if commits and 'representativeness_score' in commits[0]:
            cluster_scores = [c.get('representativeness_score', 0) for c in commits]
            all_scores.extend(cluster_scores)
            cluster_avg_scores.append(np.mean(cluster_scores))
    
    if all_scores:
        print(f"总聚类数: {total_clusters}")
        print(f"总commit数: {total_commits}")
        print(f"需要排序的聚类数: {ranked_clusters}")
        print(f"代表性得分统计:")
        print(f"  平均值: {np.mean(all_scores):.3f}")
        print(f"  中位数: {np.median(all_scores):.3f}")
        print(f"  最高分: {np.max(all_scores):.3f}")
        print(f"  最低分: {np.min(all_scores):.3f}")
        print(f"  标准差: {np.std(all_scores):.3f}")
    
    # 显示每个聚类的最具代表性commit
    print(f"\n=== 各聚类最具代表性commit ===")
    for cluster in clusters[:10]:  # 只显示前10个聚类
        cluster_id = cluster.get('cluster_id', 'unknown')
        commits = cluster.get('commits', [])
        if commits:
            most_rep = commits[0]  # 第一个就是最具代表性的
            repo_name = most_rep.get('repo_name', 'unknown')
            commit_hash = most_rep.get('hash', 'unknown')[:8]
            score = most_rep.get('representativeness_score', 0)
            summary = most_rep.get('optimization_summary_final', '')[:100]
            
            print(f"聚类 {cluster_id} (大小:{len(commits)}): {repo_name}:{commit_hash} "
                  f"(得分:{score:.3f}) - {summary}...")

def main():
    print(f"=== Commit聚类内排序分析 ===")
    print(f"输入文件: {INPUT_FILE}")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    try:
        # 1. 加载聚类结果
        clustering_data = load_clustering_results(INPUT_FILE)
        
        # 2. 处理所有聚类，进行内部排序
        ranked_data = process_clusters(clustering_data)
        
        # 输出缓存统计
        cache_stats = embedding_cache.get_cache_stats()
        print(f"\n缓存统计: 命中率 {cache_stats['hit_rate']:.2%} "
              f"(命中: {cache_stats['hit_count']}, 未命中: {cache_stats['miss_count']})")
        
        # 3. 分析排序结果
        analyze_ranking_results(ranked_data)
        
        # 4. 保存排序后的结果
        output_path = save_ranked_results(ranked_data, INPUT_FILE)
        
        if output_path:
            print(f"\n=== 排序完成 ===")
            print(f"输出文件: {output_path}")
        else:
            print(f"\n排序失败！")
            
    except Exception as e:
        print(f"\n[错误] 处理失败: {str(e)}")
        return

if __name__ == "__main__":
    main()