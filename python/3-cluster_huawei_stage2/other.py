import json
import os
import shutil
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")
SOURCE_FILE = "huawei.json"
TARGET_FILE = "summary_huawei.json"

def copy_huawei_file():
    """
    遍历所有代码库，将 huawei.json 文件复制到 summary_huawei.json 文件中。
    如果目标文件已存在则直接覆盖。
    """
    print("=" * 80)
    print("复制 huawei.json 到 summary_huawei.json")
    print("=" * 80)
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"知识库路径: {KNOWLEDGE_BASE_PATH}")
    print(f"源文件: {SOURCE_FILE}")
    print(f"目标文件: {TARGET_FILE}")
    print("-" * 80)
    
    # 检查 knowledge_base 目录是否存在
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"错误：目录 {KNOWLEDGE_BASE_PATH} 不存在")
        return
    
    # 读取代码库列表
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件不存在 - {REPO_LIST_FILE}")
        return
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repo_list = json.load(f)
    
    # 获取代码库名称列表
    repositories = []
    for repo in repo_list:
        repo_name = repo.get('name_long') or repo.get('name')
        if repo_name:
            repositories.append(repo_name)
    
    if not repositories:
        print("错误：未找到任何代码库")
        return
    
    print(f"发现 {len(repositories)} 个代码库")
    print("\n开始复制文件...")
    
    stats = {
        'total': 0,
        'success': 0,
        'skipped_not_found': 0,
        'failed': 0
    }
    
    # 遍历所有代码库
    for repo_name in tqdm(repositories, desc="复制进度", unit="repo"):
        repo_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name)
        
        # 仅处理目录（代码库）
        if not os.path.isdir(repo_path):
            continue
        
        src_file = os.path.join(repo_path, SOURCE_FILE)
        dst_file = os.path.join(repo_path, TARGET_FILE)
        
        stats['total'] += 1
        
        # 检查源文件是否存在
        if not os.path.exists(src_file):
            stats['skipped_not_found'] += 1
            continue
        
        try:
            # 复制文件，shutil.copyfile 会覆盖已经存在的目标文件
            shutil.copyfile(src_file, dst_file)
            stats['success'] += 1
        except Exception as e:
            print(f"\n[错误] 复制 {src_file} 到 {dst_file} 时出错: {e}")
            stats['failed'] += 1
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print("复制完成 - 统计信息")
    print("=" * 80)
    print(f"总代码库数: {stats['total']}")
    print(f"成功复制: {stats['success']}")
    print(f"跳过（源文件不存在）: {stats['skipped_not_found']}")
    print(f"失败: {stats['failed']}")
    
    if stats['success'] > 0:
        success_rate = (stats['success'] / stats['total']) * 100
        print(f"成功率: {success_rate:.1f}%")
    
    print("\n处理完成！")

if __name__ == "__main__":
    copy_huawei_file()

