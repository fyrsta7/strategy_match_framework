import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 读取 JSON 文件数据
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # 分别加载 repo_list_1003.json 和 repo_list_1870.json
    repos_1003 = load_json(config.root_path + "repo_list_1003.json")
    repos_1870 = load_json(config.root_path + "repo_list_1870.json")
    
    # 将repo_list_1003的代码库名称保存到集合中，便于快速查重
    https_1003 = {repo["http_url"] for repo in repos_1003}
    
    # 从repo_list_1870中筛选出不在repo_list_1003中的代码库
    diff_repos = [repo for repo in repos_1870 if repo["http_url"] not in https_1003]
    
    # 将结果写入到新的 JSON 文件
    with open(config.root_path + "repo_list_1870-1003.json", "w", encoding="utf-8") as f:
        json.dump(diff_repos, f, indent=4, ensure_ascii=False)
    
    print(f"提取完成，共提取到 {len(diff_repos)} 个代码库信息。")

if __name__ == "__main__":
    main()
