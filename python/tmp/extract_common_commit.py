"""
该脚本比对两个指定的JSON文件，提取共同出现的commit并保存到新文件中。
同时，根据提取结果保留或删除对应的文件夹。

比对规则：
- 如果commit的hash在两个文件中都存在，则认为是相同的commit
- 提取的commit将保持原有结构，不做内容修改
"""
import os
import json
import shutil
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def extract_common_commits(source_path, reference_path, output_path, modified_files_dir):
    """
    比对两个JSON文件中的commit，提取共同出现的commit到新的文件中，
    并根据结果保留或删除修改文件目录下的对应文件夹

    Args:
        source_path (str): 源文件路径，提取的commit将从这里获取
        reference_path (str): 参考文件路径，用于检查commit是否存在
        output_path (str): 输出文件路径，用于保存共同的commit
        modified_files_dir (str): 包含所有commit修改文件的目录路径

    Returns:
        tuple: (共同commit数量, 源文件总commit数量, 参考文件总commit数量, 保留文件夹数量, 删除文件夹数量)
    """
    print(f"正在处理文件...")
    print(f"- 源文件: {source_path}")
    print(f"- 参考文件: {reference_path}")
    print(f"- 输出文件: {output_path}")
    print(f"- 修改文件目录: {modified_files_dir}")

    # 确保源文件和参考文件存在
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"源文件不存在: {source_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"参考文件不存在: {reference_path}")
    if not os.path.exists(modified_files_dir):
        raise FileNotFoundError(f"修改文件目录不存在: {modified_files_dir}")

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载源文件
    with open(source_path, 'r', encoding='utf-8') as f:
        source_commits = json.load(f)
    print(f"源文件加载完成，包含 {len(source_commits)} 个commit")

    # 加载参考文件
    with open(reference_path, 'r', encoding='utf-8') as f:
        reference_commits = json.load(f)
    print(f"参考文件加载完成，包含 {len(reference_commits)} 个commit")

    # 创建参考文件中commit hash的集合，用于快速查找
    reference_hashes = {commit.get('hash', '').strip() for commit in reference_commits}
    print(f"参考文件中包含 {len(reference_hashes)} 个唯一commit hash")

    # 筛选出在参考文件中存在的commit
    common_commits = []
    common_hashes = set()
    for commit in source_commits:
        commit_hash = commit.get('hash', '').strip()
        if commit_hash and commit_hash in reference_hashes:
            common_commits.append(commit)
            common_hashes.add(commit_hash)

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(common_commits, f, indent=4, ensure_ascii=False)

    # 清理文件夹
    kept_folders = 0
    deleted_folders = 0
    
    # 遍历修改文件目录下的所有子文件夹
    for item in os.listdir(modified_files_dir):
        item_path = os.path.join(modified_files_dir, item)
        if os.path.isdir(item_path):
            # 如果文件夹名（commit hash）在共同commits中保留，否则删除
            if item in common_hashes:
                kept_folders += 1
            else:
                shutil.rmtree(item_path)
                deleted_folders += 1

    print(f"\n处理完成!")
    print(f"- 共同出现的commit: {len(common_commits)}")
    print(f"- 占源文件比例: {len(common_commits)/len(source_commits)*100:.2f}%")
    print(f"- 占参考文件比例: {len(common_commits)/len(reference_commits)*100:.2f}%")
    print(f"- 保留的文件夹数量: {kept_folders}")
    print(f"- 删除的文件夹数量: {deleted_folders}")
    print(f"- 结果已保存至: {output_path}")

    return (len(common_commits), len(source_commits), len(reference_commits), kept_folders, deleted_folders)


def main():
    # 定义文件路径
    source_path = config.root_path + "benchmark/rocksdb/is_opt_final.json"
    reference_path = config.root_path + "benchmark/rocksdb/filtered_test_result.json"
    output_path = config.root_path + "benchmark/rocksdb/benchmark.json"
    modified_files_dir = config.root_path + "benchmark/rocksdb/modified_file"

    try:
        # 执行文件比对、提取和文件夹清理
        extract_common_commits(source_path, reference_path, output_path, modified_files_dir)
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()