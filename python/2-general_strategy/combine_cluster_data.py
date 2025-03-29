import json
import os
import sys
from collections import defaultdict
from typing import Dict, Any, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局变量设置
# 最小聚类大小阈值
MIN_CLUSTER_SIZE = 16
INPUT_FILE_DIR = config.root_path + "python/2-general_strategy/result/kmeans_40000_dsv3_true_14000/"
INPUT_FILE_COMMIT = "order"
INPUT_FILE_CLUSTER = "order_sum_11_110_10_dsv3"

# 输入输出文件路径
INPUT_CLUSTERS_FILE = INPUT_FILE_DIR + f"{INPUT_FILE_COMMIT}.json"
INPUT_SUMMARIES_FILE = INPUT_FILE_DIR + f"{INPUT_FILE_CLUSTER}.json"
OUTPUT_FILE = INPUT_FILE_DIR + f"{INPUT_FILE_CLUSTER}_full_{MIN_CLUSTER_SIZE}.json"

def load_json_file(file_path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def save_json_file(data: Dict[str, Any], file_path: str):
    """保存JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved results to {file_path}")
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        sys.exit(1)

def combine_cluster_data():
    """合并聚类数据"""
    print(f"Loading clusters data from {INPUT_CLUSTERS_FILE}")
    clusters_data = load_json_file(INPUT_CLUSTERS_FILE)
    
    print(f"Loading summaries data from {INPUT_SUMMARIES_FILE}")
    summaries_data = load_json_file(INPUT_SUMMARIES_FILE)
    
    # 从第一个文件中提取符合大小要求的聚类
    filtered_clusters = {}
    for cluster_id, cluster_info in clusters_data.get("clusters", {}).items():
        if cluster_info.get("size", 0) >= MIN_CLUSTER_SIZE:
            filtered_clusters[cluster_id] = cluster_info
    
    # 获取第二个文件中的聚类总结
    summaries_by_id = {}
    for summary in summaries_data.get("cluster_summaries", []):
        cluster_id = summary.get("cluster_id")
        if cluster_id:
            summaries_by_id[cluster_id] = summary
    
    # 检查所有符合大小要求的聚类是否都有对应的总结
    missing_summaries = []
    for cluster_id in filtered_clusters:
        if cluster_id not in summaries_by_id:
            missing_summaries.append(cluster_id)
    
    if missing_summaries:
        print(f"Warning: {len(missing_summaries)} clusters with size >= {MIN_CLUSTER_SIZE} " 
              f"don't have corresponding summaries: {missing_summaries[:5]}" + 
              ("..." if len(missing_summaries) > 5 else ""))
    
    # 合并数据
    combined_results = {
        "metadata": {
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "total_clusters": len(filtered_clusters),
            "clusters_with_summaries": len(set(filtered_clusters.keys()) & set(summaries_by_id.keys())),
            "missing_summaries_count": len(missing_summaries)
        },
        "combined_clusters": []
    }
    
    # 构建合并的聚类数据
    for cluster_id, cluster_info in filtered_clusters.items():
        if cluster_id in summaries_by_id:
            summary_info = summaries_by_id[cluster_id]
            
            # 构建组合数据
            combined_cluster = {
                "cluster_id": cluster_id,
                "size": cluster_info["size"],
                "repositories_distribution": cluster_info.get("repositories", {}),
                "generic_ratio": cluster_info.get("generic_ratio", 0),
                "summary": {
                    "strategy_summary": summary_info.get("llm_summary", {}).get("strategy_summary", ""),
                    "code_examples": summary_info.get("llm_summary", {}).get("code_examples", []),
                    "application_conditions": summary_info.get("llm_summary", {}).get("application_conditions", [])
                },
                "commits": cluster_info.get("commits", [])
            }
            
            combined_results["combined_clusters"].append(combined_cluster)
    
    # 按聚类大小排序
    combined_results["combined_clusters"].sort(key=lambda x: x["size"], reverse=True)
    
    # 保存到输出文件
    print(f"Saving combined data for {len(combined_results['combined_clusters'])} clusters to {OUTPUT_FILE}")
    save_json_file(combined_results, OUTPUT_FILE)
    
    return combined_results

if __name__ == "__main__":
    print(f"Starting to combine cluster data with minimum size threshold: {MIN_CLUSTER_SIZE}")
    combined_data = combine_cluster_data()
    
    # 打印一些统计信息
    metadata = combined_data["metadata"]
    print(f"\nProcess completed:")
    print(f"- Found {metadata['total_clusters']} clusters with size >= {MIN_CLUSTER_SIZE}")
    print(f"- Successfully combined {metadata['clusters_with_summaries']} clusters with their summaries")
    print(f"- {metadata['missing_summaries_count']} clusters are missing summaries")