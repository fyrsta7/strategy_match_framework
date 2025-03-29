"""
代码库 Commit 同步脚本

该脚本检查 config.root_path/knowledge_base/ 下每个代码库的 is_opt_llm.json 文件，
并进行以下处理：
1. 如果代码库中没有 one_func_deduplicate.json 文件，则清空 is_opt_llm.json 文件（如果存在）
2. 如果代码库中有 one_func_deduplicate.json 文件，则确保 is_opt_llm.json 中的 commit 
   也存在于 one_func_deduplicate.json 中，否则删除不匹配的 commit
"""
import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from tqdm import tqdm
import concurrent.futures

def sync_repo_commits(repo_path, source_file="one_func_deduplicate.json", target_file="is_opt_llm.json"):
    """
    同步一个代码库的 commit，确保 target_file 中的 commit 也存在于 source_file 中。
    如果 source_file 不存在，则清空 target_file。
    
    Args:
        repo_path (str): 代码库路径
        source_file (str): 源文件名（commit 的基准文件）
        target_file (str): 目标文件名（需要过滤的文件）
        
    Returns:
        tuple: (代码库名, 删除的 commit 数量, 保留的 commit 数量, 状态)
    """
    repo_name = os.path.basename(repo_path)
    source_path = os.path.join(repo_path, source_file)
    target_path = os.path.join(repo_path, target_file)
    
    # 检查源文件是否存在
    if not os.path.exists(source_path):
        # 源文件不存在，检查目标文件是否存在
        if os.path.exists(target_path):
            # 目标文件存在，清空它
            try:
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
                return repo_name, -1, 0, f"源文件 {source_file} 不存在，已清空目标文件 {target_file}"
            except Exception as e:
                return repo_name, 0, 0, f"清空目标文件时出错: {str(e)}"
        else:
            return repo_name, 0, 0, f"源文件 {source_file} 不存在，目标文件 {target_file} 也不存在"
    
    # 源文件存在，但目标文件不存在
    if not os.path.exists(target_path):
        return repo_name, 0, 0, f"目标文件 {target_file} 不存在"
    
    try:
        # 读取源文件和目标文件
        with open(source_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        
        with open(target_path, 'r', encoding='utf-8') as f:
            target_data = json.load(f)
        
        # 创建源文件中所有 commit 哈希的集合（用于快速查找）
        source_hashes = {commit.get('hash', '').strip() for commit in source_data if 'hash' in commit}
        
        # 筛选目标文件中 commit，只保留在源文件中存在的 commit
        original_count = len(target_data)
        filtered_data = [commit for commit in target_data 
                        if 'hash' in commit and commit.get('hash', '').strip() in source_hashes]
        removed_count = original_count - len(filtered_data)
        
        # 只有在有 commit 被移除时才写入文件
        if removed_count > 0:
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            status = f"已删除 {removed_count} 个不匹配的 commit"
        else:
            status = "没有需要删除的 commit"
        
        return repo_name, removed_count, len(filtered_data), status
        
    except Exception as e:
        return repo_name, 0, 0, f"处理错误: {str(e)}"

def main():
    # 获取知识库根目录
    base_dir = os.path.join(config.root_path, "knowledge_base")
    
    # 确保目录存在
    if not os.path.exists(base_dir):
        print(f"错误: 目录不存在: {base_dir}")
        return
    
    # 获取所有代码库目录
    repos = [os.path.join(base_dir, repo) for repo in os.listdir(base_dir) 
             if os.path.isdir(os.path.join(base_dir, repo))]
    
    print(f"开始同步 {len(repos)} 个代码库的 commit...")
    
    # 源文件和目标文件名称
    source_file = "one_func_deduplicate.json"
    target_file = "is_opt_llm.json"
    
    # 动态确定线程数
    cpu_count = os.cpu_count()
    max_workers = 208
    
    print(f"使用 {max_workers} 个线程并行处理")
    
    # 使用并行处理加速
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建所有任务
        future_to_repo = {
            executor.submit(sync_repo_commits, repo, source_file, target_file): repo 
            for repo in repos
        }
        
        # 使用 tqdm 显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_repo), 
                          total=len(repos), 
                          desc="同步进度"):
            try:
                results.append(future.result())
            except Exception as e:
                repo = future_to_repo[future]
                print(f"处理 {os.path.basename(repo)} 时出错: {e}")
    
    # 按代码库名称排序结果
    results.sort(key=lambda x: x[0])
    
    # 输出汇总信息
    cleared_repos = sum(1 for result in results if result[1] == -1)  # 特殊值 -1 表示清空
    results_for_stats = [result for result in results if result[1] != -1]  # 排除清空的库
    total_removed = sum(result[1] for result in results_for_stats)
    total_remaining = sum(result[2] for result in results_for_stats)
    
    print("\n同步结果:")
    print(f"总共处理了 {len(results)} 个代码库")
    print(f"由于缺少源文件而清空目标文件的代码库: {cleared_repos} 个")
    print(f"删除的 commit 总数: {total_removed}")
    print(f"保留的 commit 总数: {total_remaining}")
    
    # 输出详细结果
    print("\n详细结果:")
    for repo_name, removed, remaining, status in results:
        if removed > 0 or removed == -1:  # -1 表示清空
            print(f"{repo_name}: {status}")
    
    # 输出报错信息
    errors = [result for result in results if "错误" in result[3]]
    if errors:
        print("\n处理中遇到的错误:")
        for repo_name, _, _, status in errors:
            print(f"{repo_name}: {status}")

if __name__ == "__main__":
    main()