import os
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")
SOURCE_FILENAME = "one_func.json"
OUTPUT_FILENAME = "huawei.json"
SUMMARY_OUTPUT_FILE = os.path.join(config.root_path, "all_huawei.json")

# 采样配置
SAMPLE_SIZE = 40000  # 总采样数量（测试时用10，正式运行时可改为更大值）
RANDOM_SEED = 42  # 随机种子
MAX_WORKERS = 128  # 并行线程数

# 全局统计锁
stats_lock = threading.Lock()
global_stats = {
    'total_repos': 0,
    'repos_with_commits': 0,
    'total_commits_before_sampling': 0,
    'total_commits_sampled': 0,
    'repos_sampled': 0
}

def load_repository_commits(repo_name):
    """
    加载单个代码库的one_func.json
    返回: (repo_name, commits_list, error_message)
    """
    json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, SOURCE_FILENAME)
    
    if not os.path.exists(json_file_path):
        return repo_name, [], None
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
        
        if not isinstance(commits, list):
            return repo_name, [], "JSON is not a list"
        
        # 为每个commit添加repository_name字段（如果没有）
        for commit in commits:
            if 'repository_name' not in commit:
                commit['repository_name'] = repo_name
        
        return repo_name, commits, None
        
    except json.JSONDecodeError as e:
        return repo_name, [], f"JSON decode error: {str(e)}"
    except Exception as e:
        return repo_name, [], f"Error: {str(e)}"

def save_repository_commits(repo_name, commits):
    """
    保存采样后的commits到huawei.json
    返回: 错误信息（如果有）
    """
    output_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, OUTPUT_FILENAME)
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(commits, f, ensure_ascii=False, indent=2)
        
        return None
        
    except Exception as e:
        return f"Error saving commits: {str(e)}"

def main():
    """主函数"""
    print("=" * 80)
    print("从 one_func.json 中随机采样commits")
    print("=" * 80)
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"知识库路径: {KNOWLEDGE_BASE_PATH}")
    print(f"源文件名: {SOURCE_FILENAME}")
    print(f"输出文件名: {OUTPUT_FILENAME}")
    print(f"采样策略: 总量控制（从所有代码库共采样 {SAMPLE_SIZE} 个commits）")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"并行线程数: {MAX_WORKERS}")
    print("-" * 80)
    
    # 设置随机种子
    random.seed(RANDOM_SEED)
    
    # 读取代码库列表
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件不存在 - {REPO_LIST_FILE}")
        return
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repo_list = json.load(f)
    
    # 获取代码库名称列表
    repositories = []
    for repo in repo_list:
        # 优先使用name_long，如果没有则使用name
        repo_name = repo.get('name_long') or repo.get('name')
        if repo_name:
            repositories.append(repo_name)
    
    if not repositories:
        print("错误：未找到任何代码库")
        return
    
    print(f"发现 {len(repositories)} 个代码库")
    print("\n第一步：并行加载所有代码库的commits...")
    
    # 并行加载所有代码库的commits
    all_commits_with_repo = []  # 存储 (commit, repo_name) 元组
    repo_commit_counts = {}  # 存储每个代码库的commit数量
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_repo = {executor.submit(load_repository_commits, repo): repo for repo in repositories}
        
        # 创建进度条
        pbar = tqdm(total=len(repositories), desc="加载代码库", unit="repo")
        
        for future in as_completed(future_to_repo):
            repo_name = future_to_repo[future]
            try:
                repo_name_result, commits, error = future.result()
                
                global_stats['total_repos'] += 1
                
                if error:
                    print(f"\n[警告] 代码库 {repo_name}: {error}")
                    repo_commit_counts[repo_name] = 0
                elif commits:
                    global_stats['repos_with_commits'] += 1
                    global_stats['total_commits_before_sampling'] += len(commits)
                    repo_commit_counts[repo_name] = len(commits)
                    
                    # 存储每个commit及其所属代码库
                    for commit in commits:
                        all_commits_with_repo.append((commit, repo_name))
                else:
                    repo_commit_counts[repo_name] = 0
                
                pbar.set_postfix({
                    "加载commits": global_stats['total_commits_before_sampling']
                })
                
            except Exception as e:
                print(f"\n[异常] 代码库 {repo_name}: {str(e)}")
                repo_commit_counts[repo_name] = 0
            
            pbar.update(1)
        
        pbar.close()
    
    print(f"\n加载完成：")
    print(f"  有效代码库数: {global_stats['repos_with_commits']}")
    print(f"  总commit数: {global_stats['total_commits_before_sampling']}")
    
    if not all_commits_with_repo:
        print("\n错误：未找到任何commits")
        return
    
    # 第二步：随机采样
    print(f"\n第二步：从 {global_stats['total_commits_before_sampling']} 个commits中随机采样 {SAMPLE_SIZE} 个...")
    
    if len(all_commits_with_repo) <= SAMPLE_SIZE:
        print(f"[提示] 可用commits数量({len(all_commits_with_repo)})少于或等于采样数量({SAMPLE_SIZE})，将全部采样")
        sampled_commits_with_repo = all_commits_with_repo
    else:
        sampled_commits_with_repo = random.sample(all_commits_with_repo, SAMPLE_SIZE)
    
    global_stats['total_commits_sampled'] = len(sampled_commits_with_repo)
    
    # 第三步：按代码库分组
    print(f"\n第三步：按代码库分组采样结果...")
    repo_sampled_commits = {}
    for commit, repo_name in sampled_commits_with_repo:
        if repo_name not in repo_sampled_commits:
            repo_sampled_commits[repo_name] = []
        repo_sampled_commits[repo_name].append(commit)
    
    global_stats['repos_sampled'] = len(repo_sampled_commits)
    
    # 第四步：保存到各代码库的huawei.json
    print(f"\n第四步：保存到各代码库的 {OUTPUT_FILENAME}...")
    save_errors = []
    
    for repo_name, commits in tqdm(repo_sampled_commits.items(), desc="保存到代码库", unit="repo"):
        error = save_repository_commits(repo_name, commits)
        if error:
            save_errors.append((repo_name, error))
            print(f"\n[错误] 保存 {repo_name} 失败: {error}")
    
    # 第五步：汇总到all_huawei.json
    print(f"\n第五步：汇总到 {SUMMARY_OUTPUT_FILE}...")
    all_huawei_commits = [commit for commit, _ in sampled_commits_with_repo]
    
    try:
        with open(SUMMARY_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_huawei_commits, f, ensure_ascii=False, indent=2)
        print(f"✓ 成功保存 {len(all_huawei_commits)} 个commits到 {SUMMARY_OUTPUT_FILE}")
    except Exception as e:
        print(f"✗ 保存汇总文件失败: {str(e)}")
        return
    
    # 输出最终统计
    print("\n" + "=" * 80)
    print("采样完成 - 最终统计")
    print("=" * 80)
    print(f"代码库统计:")
    print(f"  总代码库数: {global_stats['total_repos']}")
    print(f"  有commits的代码库数: {global_stats['repos_with_commits']}")
    print(f"  被采样涉及的代码库数: {global_stats['repos_sampled']}")
    print(f"\nCommit统计:")
    print(f"  采样前总数: {global_stats['total_commits_before_sampling']}")
    print(f"  采样后总数: {global_stats['total_commits_sampled']}")
    print(f"  采样率: {(global_stats['total_commits_sampled'] / global_stats['total_commits_before_sampling'] * 100):.2f}%")
    
    print(f"\n各代码库采样分布:")
    for repo_name in sorted(repo_sampled_commits.keys()):
        count = len(repo_sampled_commits[repo_name])
        original_count = repo_commit_counts.get(repo_name, 0)
        print(f"  {repo_name}: {count} 个 (原始: {original_count} 个)")
    
    if save_errors:
        print(f"\n保存错误 ({len(save_errors)} 个):")
        for repo_name, error in save_errors:
            print(f"  {repo_name}: {error}")
    
    print("\n处理完成！")

if __name__ == "__main__":
    main()

