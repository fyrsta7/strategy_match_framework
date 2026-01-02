import json
import os
import shutil
from git import Repo
from tqdm import tqdm
import concurrent.futures
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def copy_opt_file():
    """
    遍历 config.root_path/knowledge_base 中的所有代码库，对于每个代码库，
    将 is_opt_final.json 文件复制到 summary.json 文件中。如果目标文件已存在则直接覆盖。
    """
    knowledge_base_path = os.path.join(config.root_path, "knowledge_base_all")
    
    # 检查 knowledge_base 目录是否存在
    if not os.path.exists(knowledge_base_path):
        print(f"目录 {knowledge_base_path} 不存在")
        return
    
    for repo_name in os.listdir(knowledge_base_path):
        repo_path = os.path.join(knowledge_base_path, repo_name)
        # 仅处理目录（代码库）
        if os.path.isdir(repo_path):
            src_file = os.path.join(repo_path, "is_opt_final.json")
            dst_file = os.path.join(repo_path, "summary.json")
            
            # 检查源文件是否存在
            if os.path.exists(src_file):
                try:
                    # 复制文件，shutil.copyfile 会覆盖已经存在的目标文件
                    shutil.copyfile(src_file, dst_file)
                    print(f"成功复制文件:\n  {src_file}\n到\n  {dst_file}")
                except Exception as e:
                    print(f"复制 {src_file} 到 {dst_file} 时出错: {e}")
            else:
                print(f"文件 {src_file} 不存在，跳过仓库：{repo_name}")



copy_opt_file()