import os
import json
import shutil
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def process_modified_files():
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 定义文件路径
    json_path = os.path.join(config.root_path, 'benchmark/rocksdb/filtered_test_result.json')
    modified_files_dir = os.path.join(config.root_path, 'benchmark/rocksdb/modified_file')
    
    # 检查文件和目录是否存在
    if not os.path.exists(json_path):
        logging.error(f"JSON file not found: {json_path}")
        return
    
    if not os.path.exists(modified_files_dir):
        logging.error(f"Modified files directory not found: {modified_files_dir}")
        return
    
    # 读取JSON文件
    try:
        with open(json_path, 'r') as f:
            commit_data = json.load(f)
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")
        return
    
    # 提取需要保留的commit hash
    valid_commits = {commit['hash'] for commit in commit_data}
    logging.info(f"Found {len(valid_commits)} valid commits in JSON file")
    
    # 获取modified_file目录下所有子文件夹（即commit hash文件夹）
    try:
        existing_commit_dirs = [d for d in os.listdir(modified_files_dir) 
                               if os.path.isdir(os.path.join(modified_files_dir, d))]
    except Exception as e:
        logging.error(f"Error listing directories: {e}")
        return
    
    # 删除不在valid_commits中的文件夹
    for commit_dir in existing_commit_dirs:
        if commit_dir not in valid_commits:
            dir_to_remove = os.path.join(modified_files_dir, commit_dir)
            try:
                shutil.rmtree(dir_to_remove)
                logging.info(f"Removed directory for invalid commit: {commit_dir}")
            except Exception as e:
                logging.error(f"Error removing directory {dir_to_remove}: {e}")
    
    # 检查是否所有valid_commits都有对应文件夹
    missing_dirs = [commit for commit in valid_commits 
                    if not os.path.exists(os.path.join(modified_files_dir, commit))]
    
    if missing_dirs:
        logging.warning(f"Missing directories for {len(missing_dirs)} valid commits")
        for commit in missing_dirs[:10]:  # 只显示前10个，避免日志过长
            logging.warning(f"Missing directory for commit: {commit}")
        if len(missing_dirs) > 10:
            logging.warning(f"... and {len(missing_dirs) - 10} more")
    
    logging.info("Processing completed")

if __name__ == "__main__":
    process_modified_files()