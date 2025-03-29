import os
import json
import csv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from tqdm import tqdm
import concurrent.futures

def count_commits_in_repo(repo_path, json_files):
    """
    统计一个代码库中指定的几个 JSON 文件包含的 commit 数量
    
    Args:
        repo_path (str): 代码库路径
        json_files (list): 要统计的 JSON 文件名列表
        
    Returns:
        dict: 包含每个 JSON 文件 commit 数量的字典
    """
    results = {'repo_name': os.path.basename(repo_path)}
    
    for json_file in json_files:
        file_path = os.path.join(repo_path, json_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                commit_count = len(data)
            except Exception as e:
                commit_count = f"错误: {str(e)}"
        else:
            commit_count = "文件不存在"
            
        # 去掉 .json 后缀作为字典的键
        key = json_file.replace('.json', '')
        results[key] = commit_count
        
    return results

def main():
    """
    主函数：遍历 knowledge_base 目录下所有代码库，
    统计指定 JSON 文件的 commit 数量并输出到 CSV 文件
    """
    # 要统计的 JSON 文件列表（按要求顺序）
    json_files = [
        'all_commit.json',
        'one_file.json',
        'c_language.json',
        'is_opt_keyword.json',
        'has_file.json',
        'has_file_with_func.json',
        'one_func.json',
        'one_func_deduplicate.json',
        'is_opt_llm.json',
        'is_opt_final.json'
    ]
    
    # 获取知识库根目录
    base_dir = os.path.join(config.root_path, "knowledge_base")
    
    # 确保目录存在
    if not os.path.exists(base_dir):
        print(f"错误: 目录不存在: {base_dir}")
        return
    
    # 获取所有代码库目录
    repos = [os.path.join(base_dir, repo) for repo in os.listdir(base_dir) 
             if os.path.isdir(os.path.join(base_dir, repo))]
    
    print(f"开始统计 {len(repos)} 个代码库的 commit 数量...")
    
    # 设置最大线程数为24（可以根据需要调整）
    max_workers = 208
    print(f"使用并行处理，最大线程数: {max_workers}")
    
    # 使用并行处理加速，并明确指定最大线程数
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建所有任务
        future_to_repo = {
            executor.submit(count_commits_in_repo, repo, json_files): repo 
            for repo in repos
        }
        
        # 使用 tqdm 显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_repo), 
                          total=len(repos), 
                          desc="处理进度"):
            try:
                results.append(future.result())
            except Exception as e:
                repo = future_to_repo[future]
                print(f"处理 {os.path.basename(repo)} 时出错: {e}")
    
    # 输出结果到 CSV 文件
    csv_file = 'commit_statistics.csv'
    fieldnames = ['repo_name'] + [file.replace('.json', '') for file in json_files]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda x: x['repo_name']):
            writer.writerow(result)
    
    # 打印总结信息
    print(f"\n统计完成! 结果已保存到 {csv_file}")
    
    # 显示一些汇总数据
    print("\n总结:")
    for json_file in json_files:
        key = json_file.replace('.json', '')
        valid_counts = [r[key] for r in results if isinstance(r[key], int)]
        if valid_counts:
            total = sum(valid_counts)
            avg = total / len(valid_counts)
            print(f"{key}: 总数 = {total}, 平均每个库 = {avg:.2f}, 有效库数量 = {len(valid_counts)}")
        else:
            print(f"{key}: 没有有效数据")

if __name__ == "__main__":
    main()