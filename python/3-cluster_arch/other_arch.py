import json
import os
import shutil
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
INPUT_FILE_NAME = "is_opt_arch_final.json"
OUTPUT_FILE_NAME = "summary_arch.json"

def copy_opt_file():
    """
    遍历 KNOWLEDGE_BASE_PATH 中的所有代码库，对于每个代码库，
    将 is_opt_arch_final.json 文件复制到 summary_arch.json 文件中。
    如果目标文件已存在则直接覆盖。
    """
    
    # 读取代码库列表
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件不存在 - {REPO_LIST_FILE}")
        return
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repos = json.load(f)
    
    print(f"从 {REPO_LIST_FILE} 读取到 {len(repos)} 个代码库")
    
    # 检查 knowledge_base 目录是否存在
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"目录 {KNOWLEDGE_BASE_PATH} 不存在")
        return
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for repo in tqdm(repos, desc="复制文件"):
        repo_name = repo.get('name_long', repo.get('name', ''))
        if not repo_name:
            print(f"警告：跳过无名称的代码库")
            error_count += 1
            continue
            
        repo_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name)
        
        # 仅处理目录（代码库）
        if os.path.isdir(repo_path):
            src_file = os.path.join(repo_path, INPUT_FILE_NAME)
            dst_file = os.path.join(repo_path, OUTPUT_FILE_NAME)
            
            # 检查源文件是否存在
            if os.path.exists(src_file):
                try:
                    # 复制文件，shutil.copyfile 会覆盖已经存在的目标文件
                    shutil.copyfile(src_file, dst_file)
                    success_count += 1
                except Exception as e:
                    print(f"\n复制 {src_file} 到 {dst_file} 时出错: {e}")
                    error_count += 1
            else:
                skip_count += 1
        else:
            skip_count += 1
    
    print(f"\n=== 复制结果 ===")
    print(f"成功复制: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"错误: {error_count}")
    print(f"总计: {len(repos)}")

if __name__ == "__main__":
    copy_opt_file()

