import os
import shutil
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def delete_bm25_folders(root_path):
    # 构建benchmark/rocksdb/modified_file路径
    base_path = os.path.join(root_path, "benchmark", "rocksdb", "modified_file")
    
    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"路径不存在: {base_path}")
        return False
    
    # 计数器
    deleted_count = 0
    
    # 遍历所有commit哈希文件夹
    for commit_hash in os.listdir(base_path):
        commit_path = os.path.join(base_path, commit_hash)
        
        # 检查是否是目录
        if not os.path.isdir(commit_path):
            continue
        
        # 检查bm25目录是否存在
        bm25_path = os.path.join(commit_path, "bm25")
        if os.path.exists(bm25_path) and os.path.isdir(bm25_path):
            try:
                # 删除bm25目录
                shutil.rmtree(bm25_path)
                print(f"已删除: {bm25_path}")
                deleted_count += 1
            except Exception as e:
                print(f"删除 {bm25_path} 时出错: {e}")
    
    print(f"总共删除了 {deleted_count} 个bm25文件夹")
    return True

if __name__ == "__main__":
    success = delete_bm25_folders(config.root_path)