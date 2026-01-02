#!/usr/bin/env python3
"""
Merge cluster files: combine existing clusters with randomly sampled clusters.
This script merges 316 existing clusters with 700 randomly selected clusters
from a larger pool, ensuring no duplicate commits.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

import json
import random
from typing import Dict, List, Set, Any


# ============ 配置变量 ============
# 现有聚类文件（316个聚类）
EXISTING_FILE = os.path.join(
    config.root_path,
    "python/3-cluster_huawei_stage2/result_commit_10002/0_8_2.json"
)

# 候选聚类文件（2983个聚类）
CANDIDATE_FILE = os.path.join(
    config.root_path,
    "python/3-cluster/result_30342/0_8_2_True.json"
)

# 输出文件
OUTPUT_FILE = os.path.join(
    config.root_path,
    "python/3-cluster_huawei_stage2/result_final/0_8_2_merged.json"
)

# 采样配置
NUM_SAMPLES = 700          # 从候选池中采样的聚类数量
MIN_CLUSTER_SIZE = 2       # 最小聚类大小
MAX_CLUSTER_SIZE = 4       # 最大聚类大小
RANDOM_SEED = None         # 随机种子（None表示不固定种子，设置数字如42可固定结果）


# ============ 核心函数 ============
def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    print(f"Loading {os.path.basename(file_path)}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data.get('clusters', []))} clusters")
    return data


def extract_commit_hashes(clusters: List[Dict]) -> Set[str]:
    """Extract all commit hashes from a list of clusters."""
    commit_hashes = set()
    for cluster in clusters:
        for commit in cluster.get('commits', []):
            commit_hashes.add(commit['hash'])
    return commit_hashes


def normalize_commit_fields(commit: Dict) -> Dict:
    """
    Normalize commit fields to ensure compatibility with order.py.
    Converts 'optimization_summary_final' to 'optimization_summary_huawei_final' if needed.
    """
    # If the commit already has optimization_summary_huawei_final, return as is
    if 'optimization_summary_huawei_final' in commit:
        return commit
    
    # If it has optimization_summary_final, copy it to optimization_summary_huawei_final
    if 'optimization_summary_final' in commit:
        commit['optimization_summary_huawei_final'] = commit['optimization_summary_final']
    
    return commit


def filter_candidate_clusters(
    clusters: List[Dict],
    existing_commits: Set[str],
    min_size: int = 2,
    max_size: int = 4
) -> List[Dict]:
    """
    Filter clusters based on size and commit uniqueness.
    Also normalizes commit fields to ensure compatibility with order.py.
    
    Args:
        clusters: List of all candidate clusters
        existing_commits: Set of commit hashes that already exist
        min_size: Minimum cluster size (inclusive)
        max_size: Maximum cluster size (inclusive)
    
    Returns:
        List of filtered clusters that meet all criteria
    """
    print(f"Filtering candidate clusters (size: {min_size}-{max_size}, no duplicate commits)...")
    
    valid_clusters = []
    for cluster in clusters:
        size = cluster.get('size', 0)
        
        # Check size constraint
        if size < min_size or size > max_size:
            continue
        
        # Check if any commit already exists
        cluster_commits = {commit['hash'] for commit in cluster.get('commits', [])}
        if cluster_commits.intersection(existing_commits):
            continue
        
        # Normalize commit fields for compatibility with order.py
        for commit in cluster.get('commits', []):
            normalize_commit_fields(commit)
        
        valid_clusters.append(cluster)
    
    print(f"  ✓ Found {len(valid_clusters)} valid candidate clusters")
    return valid_clusters


def sample_clusters(clusters: List[Dict], n: int, seed: int = None) -> List[Dict]:
    """
    Randomly sample n clusters from the candidate pool.
    
    Args:
        clusters: List of candidate clusters
        n: Number of clusters to sample
        seed: Random seed for reproducibility (optional)
    
    Returns:
        List of randomly sampled clusters
    """
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    if len(clusters) < n:
        print(f"⚠️  WARNING: Only {len(clusters)} clusters available, less than requested {n}")
        print(f"  ✓ Using all {len(clusters)} available clusters")
        return clusters
    
    print(f"Randomly sampling {n} clusters from {len(clusters)} candidates...")
    sampled = random.sample(clusters, n)
    print(f"  ✓ Successfully sampled {len(sampled)} clusters")
    return sampled


def merge_and_sort_clusters(
    existing_clusters: List[Dict],
    new_clusters: List[Dict]
) -> List[Dict]:
    """
    Merge two lists of clusters and sort by size (descending).
    Also reassign cluster IDs starting from 1.
    
    Args:
        existing_clusters: List of existing clusters
        new_clusters: List of newly sampled clusters
    
    Returns:
        Merged and sorted list of clusters with reassigned IDs
    """
    print(f"Merging clusters...")
    print(f"  - Existing clusters: {len(existing_clusters)}")
    print(f"  - New clusters: {len(new_clusters)}")
    
    # Merge
    all_clusters = existing_clusters + new_clusters
    print(f"  - Total clusters: {len(all_clusters)}")
    
    # Sort by size (descending)
    print(f"  - Sorting by size (descending)...")
    all_clusters.sort(key=lambda x: x.get('size', 0), reverse=True)
    
    # Reassign cluster IDs
    print(f"  - Reassigning cluster IDs...")
    for idx, cluster in enumerate(all_clusters, start=1):
        cluster['cluster_id'] = idx
    
    print(f"  ✓ Merge and sort completed")
    return all_clusters


def create_output_config(
    existing_config: Dict,
    candidate_config: Dict,
    total_clusters: int,
    existing_count: int,
    sampled_count: int
) -> Dict[str, Any]:
    """
    Create configuration section for the output file.
    Note: Only cluster information is included, no noise points.
    """
    config = {
        "source_info": {
            "existing_file": "result_commit_10002/0_8_2.json",
            "existing_clusters": existing_count,
            "candidate_file": "result_30342/0_8_2_True.json",
            "sampled_clusters": sampled_count,
            "total_merged_clusters": total_clusters,
            "note": "Only clusters are included, noise points are excluded"
        },
        "min_similarity_threshold": existing_config.get('min_similarity_threshold', 0.8),
        "algorithm": existing_config.get('algorithm', 'DBSCAN'),
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "max_cluster_size": MAX_CLUSTER_SIZE,
        "total_clusters": total_clusters
    }
    return config


def save_output(
    output_data: Dict[str, Any],
    output_file: str
) -> None:
    """Save the merged clusters to output file."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    print(f"Saving output to {os.path.basename(output_file)}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"  ✓ File saved successfully ({file_size_mb:.2f} MB)")


def main():
    print("=" * 80)
    print("📊 CLUSTER MERGE TOOL")
    print("=" * 80)
    print(f"\n配置信息:")
    print(f"  - 现有聚类文件: {EXISTING_FILE}")
    print(f"  - 候选聚类文件: {CANDIDATE_FILE}")
    print(f"  - 输出文件: {OUTPUT_FILE}")
    print(f"  - 采样数量: {NUM_SAMPLES}")
    print(f"  - 聚类大小范围: {MIN_CLUSTER_SIZE}-{MAX_CLUSTER_SIZE}")
    if RANDOM_SEED is not None:
        print(f"  - 随机种子: {RANDOM_SEED}")
    
    # Step 1: Load existing clusters
    print("\n" + "=" * 80)
    print("步骤 1: 加载现有聚类")
    print("=" * 80)
    existing_data = load_json_file(EXISTING_FILE)
    existing_clusters = existing_data.get('clusters', [])
    print(f"  - Total commits in existing clusters: {sum(c.get('size', 0) for c in existing_clusters)}")
    
    # Step 2: Extract existing commit hashes
    print("\n" + "=" * 80)
    print("步骤 2: 提取现有 commit hashes")
    print("=" * 80)
    existing_commits = extract_commit_hashes(existing_clusters)
    print(f"  - Found {len(existing_commits)} unique commits in existing clusters")
    
    # Step 3: Load candidate clusters
    print("\n" + "=" * 80)
    print("步骤 3: 加载候选聚类")
    print("=" * 80)
    candidate_data = load_json_file(CANDIDATE_FILE)
    candidate_clusters = candidate_data.get('clusters', [])
    print(f"  - Total commits in candidate clusters: {sum(c.get('size', 0) for c in candidate_clusters)}")
    
    # Step 4: Filter candidate clusters
    print("\n" + "=" * 80)
    print("步骤 4: 筛选候选聚类")
    print("=" * 80)
    valid_candidates = filter_candidate_clusters(
        candidate_clusters,
        existing_commits,
        min_size=MIN_CLUSTER_SIZE,
        max_size=MAX_CLUSTER_SIZE
    )
    
    if len(valid_candidates) == 0:
        print("\n❌ ERROR: No valid candidate clusters found!")
        print("Please check the filtering criteria.")
        return 1
    
    # Step 5: Sample clusters
    print("\n" + "=" * 80)
    print("步骤 5: 随机采样聚类")
    print("=" * 80)
    sampled_clusters = sample_clusters(valid_candidates, NUM_SAMPLES, seed=RANDOM_SEED)
    
    # Step 6: Merge and sort
    print("\n" + "=" * 80)
    print("步骤 6: 合并并排序")
    print("=" * 80)
    merged_clusters = merge_and_sort_clusters(existing_clusters, sampled_clusters)
    
    # Step 7: Create output structure
    print("\n" + "=" * 80)
    print("步骤 7: 创建输出结构")
    print("=" * 80)
    output_config = create_output_config(
        existing_data.get('clustering_config', {}),
        candidate_data.get('clustering_config', {}),
        len(merged_clusters),
        len(existing_clusters),
        len(sampled_clusters)
    )
    
    # Only output clusters, no noise points
    output_data = {
        "clustering_config": output_config,
        "clusters": merged_clusters
    }
    print(f"  ✓ Output structure created (clusters only, no noise points)")
    
    # Step 8: Save output
    print("\n" + "=" * 80)
    print("步骤 8: 保存输出文件")
    print("=" * 80)
    save_output(output_data, OUTPUT_FILE)
    
    # Final statistics
    print("\n" + "=" * 80)
    print("✅ MERGE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"📌 Existing clusters: {len(existing_clusters)}")
    print(f"📌 Sampled clusters: {len(sampled_clusters)}")
    print(f"📌 Total clusters: {len(merged_clusters)}")
    print(f"📌 Output file: {OUTPUT_FILE}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

