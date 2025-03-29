import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def deduplicate_repo_list():
    # 构造 repo_list.json 文件的完整路径
    repo_list_file = os.path.join(config.root_path, "repo_list_31308.json")
    
    # 读取 JSON 数据
    with open(repo_list_file, "r", encoding="utf-8") as f:
        repo_data = json.load(f)
    
    unique_repos = []
    seen_names = set()
    
    # 遍历原始列表，保留第一次出现的记录
    for repo in repo_data:
        repo_name = repo.get("name")
        if repo_name is None:
            # 如果没有 name 字段，可以选择跳过或者直接添加
            unique_repos.append(repo)
        elif repo_name not in seen_names:
            seen_names.add(repo_name)
            unique_repos.append(repo)
        else:
            # 如果已经存在，则跳过（删除后面的记录）
            continue
    
    # 保存去重后的结果，可选择覆盖原文件或另存为新文件
    with open(repo_list_file, "w", encoding="utf-8") as f:
        json.dump(unique_repos, f, indent=4, ensure_ascii=False)
    
    print(f"去重完成，总计 {len(repo_data)} 条记录，保留 {len(unique_repos)} 条记录。")

if __name__ == "__main__":
    deduplicate_repo_list()