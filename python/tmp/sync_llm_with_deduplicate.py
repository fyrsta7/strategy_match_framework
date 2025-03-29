"""
该脚本遍历所有代码库，检查 is_opt_llm.json 文件中的 commit 是否存在于 one_func_deduplicate.json 中。
如果某个 commit 在 is_opt_llm.json 中但不在 one_func_deduplicate.json 中，则将其从 is_opt_llm.json 中删除。
"""

import os
import json
import sys
from tqdm import tqdm
import concurrent.futures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def sync_llm_with_deduplicate(repo_name):
    """
    同步单个代码库的 is_opt_llm.json 与 one_func_deduplicate.json 文件
    
    Args:
        repo_name (str): 代码库名称
        
    Returns:
        dict: 包含处理结果的字典
    """
    result = {
        'repo_name': repo_name,
        'status': 'skipped',
        'message': '',
        'removed_count': 0
    }
    
    repo_path = os.path.join(config.root_path, "knowledge_base", repo_name)
    
    # 检查代码库路径是否存在
    if not os.path.isdir(repo_path):
        result['message'] = "代码库路径不存在"
        return result
    
    # 定义文件路径
    dedup_path = os.path.join(repo_path, "one_func_deduplicate.json")
    llm_path = os.path.join(repo_path, "is_opt_llm.json")
    
    # 检查两个文件是否都存在
    if not os.path.exists(dedup_path):
        result['message'] = "one_func_deduplicate.json 不存在"
        return result
        
    if not os.path.exists(llm_path):
        result['message'] = "is_opt_llm.json 不存在"
        return result
    
    try:
        # 读取 one_func_deduplicate.json 文件
        with open(dedup_path, 'r', encoding='utf-8') as f:
            dedup_commits = json.load(f)
        
        # 读取 is_opt_llm.json 文件
        with open(llm_path, 'r', encoding='utf-8') as f:
            llm_commits = json.load(f)
        
        # 创建 dedup_commits 中所有 commit 哈希的集合，用于快速查找
        dedup_hashes = {commit.get('hash', '') for commit in dedup_commits}
        
        # 筛选出在 dedup_commits 中存在的 commit
        retained_commits = []
        removed_count = 0
        
        for commit in llm_commits:
            commit_hash = commit.get('hash', '')
            if commit_hash in dedup_hashes:
                retained_commits.append(commit)
            else:
                removed_count += 1
        
        # 如果有需要删除的 commit，更新 is_opt_llm.json 文件
        if removed_count > 0:
            with open(llm_path, 'w', encoding='utf-8') as f:
                json.dump(retained_commits, f, indent=4, ensure_ascii=False)
            
            result['status'] = 'updated'
            result['message'] = f"已从 is_opt_llm.json 中移除 {removed_count} 个不存在于 one_func_deduplicate.json 的 commit"
        else:
            result['status'] = 'unchanged'
            result['message'] = "没有需要移除的 commit"
            
        result['removed_count'] = removed_count
        return result
        
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"处理出错: {str(e)}"
        return result

def main():
    """主函数"""
    # 获取知识库根目录
    knowledge_base_path = os.path.join(config.root_path, "knowledge_base")
    
    # 检查目录是否存在
    if not os.path.exists(knowledge_base_path):
        print(f"错误: 目录不存在: {knowledge_base_path}")
        return
    
    # 获取所有代码库目录
    repositories = [repo for repo in os.listdir(knowledge_base_path) 
                   if os.path.isdir(os.path.join(knowledge_base_path, repo))]
    
    print(f"开始处理 {len(repositories)} 个代码库...")
    
    # 初始化统计变量
    total_updated = 0
    total_removed = 0
    total_errors = 0
    
    # 设置进度条
    with tqdm(total=len(repositories), desc="同步处理") as pbar:
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # 提交所有任务
            future_to_repo = {executor.submit(sync_llm_with_deduplicate, repo): repo for repo in repositories}
            
            # 处理结果
            for future in concurrent.futures.as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    result = future.result()
                    
                    # 更新统计信息
                    if result['status'] == 'updated':
                        total_updated += 1
                        total_removed += result['removed_count']
                        tqdm.write(f"[{repo}] {result['message']}")
                    elif result['status'] == 'error':
                        total_errors += 1
                        tqdm.write(f"[{repo}] 错误: {result['message']}")
                        
                except Exception as exc:
                    total_errors += 1
                    tqdm.write(f"[{repo}] 处理异常: {str(exc)}")
                
                # 更新进度条
                pbar.update(1)
    
    # 打印总结信息
    print("\n处理完成!")
    print(f"- 更新的代码库: {total_updated}")
    print(f"- 移除的 commit 总数: {total_removed}")
    print(f"- 处理出错的代码库: {total_errors}")

if __name__ == "__main__":
    main()