import os
import sys
import json
from tqdm import tqdm  # 用于进度显示
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"读取 {file_path} 时发生错误：{e}")
        sys.exit(1)

def save_json_file(data, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"写入 {file_path} 时发生错误：{e}")
        sys.exit(1)

def process_repo(repo_name):
    print("-" * 40)
    print(f"开始处理代码库： {repo_name}")
    
    # 构造 repo 的结果目录路径
    repo_result_dir = os.path.join(config.root_path, "result", repo_name)
    
    # 定义各文件的路径
    is_opt_keyword_file = os.path.join(repo_result_dir, "is_opt_keyword.json")
    one_func_file = os.path.join(repo_result_dir, "one_func.json")
    is_opt_keyword_deepseek_file = os.path.join(repo_result_dir, "is_opt_keyword_deepseek.json")

    # 如果 is_opt_keyword_deepseek.json 不存在，则创建并复制 is_opt_keyword.json 的内容
    if not os.path.exists(is_opt_keyword_deepseek_file):
        if not os.path.exists(is_opt_keyword_file):
            print(f"  [ERROR] 源文件不存在：{is_opt_keyword_file}")
            return
        print(f"  {is_opt_keyword_deepseek_file} 不存在，开始创建并复制内容...")
        data = load_json_file(is_opt_keyword_file)
        save_json_file(data, is_opt_keyword_deepseek_file)
        print(f"  创建完成： {is_opt_keyword_deepseek_file}")
    else:
        print(f"  文件 {is_opt_keyword_deepseek_file} 已存在。")
    
    # 加载 is_opt_keyword_deepseek.json 中的 commit 列表
    commits_deepseek = load_json_file(is_opt_keyword_deepseek_file)
    
    # 尝试加载 one_func.json 中的 commit 列表
    if not os.path.exists(one_func_file):
        print(f"  警告：文件 {one_func_file} 不存在，后续所有commit将视作未在 one_func.json 中存在。")
        one_func_commit_hash_set = set()
    else:
        one_func_commits = load_json_file(one_func_file)
        one_func_commit_hash_set = { commit.get("hash") for commit in one_func_commits if commit.get("hash") }
    
    # 遍历每个 commit 检测是否需要添加 is_code_optimization 字段
    modified = False
    for commit in commits_deepseek:
        # 如果已有 is_code_optimization 字段，跳过
        if "is_code_optimization" in commit:
            continue

        commit_hash = commit.get("hash")
        if not commit_hash:
            continue

        if commit_hash not in one_func_commit_hash_set:
            commit["is_code_optimization"] = "not in one_func.json, don't need deepseek"
            modified = True
            print(f"  为 commit {commit_hash} 添加了 is_code_optimization 字段。")
    
    if modified:
        save_json_file(commits_deepseek, is_opt_keyword_deepseek_file)
        print(f"  更新后的数据已保存到 {is_opt_keyword_deepseek_file}。")
    else:
        print("  所有 commit 均不需要修改。")
    
def main():
    # 构造 repo_list 文件的路径
    repo_list_file = os.path.join(config.root_path, "repo_list_1870_1003.json")
    
    if not os.path.exists(repo_list_file):
        print(f"repo列表文件不存在：{repo_list_file}")
        sys.exit(1)
    
    # 加载仓库列表，此文件中每个对象都有 name 字段
    repo_list = load_json_file(repo_list_file)
    
    print("开始处理代码库列表...")
    # 使用 tqdm 追踪总体进度
    for repo in tqdm(repo_list, desc="处理代码库", unit="repo"):
        repo_name = repo.get("name")
        if repo_name:
            process_repo(repo_name)
        else:
            print("  检测到没有'name'字段的记录，跳过。")

if __name__ == '__main__':
    main()
