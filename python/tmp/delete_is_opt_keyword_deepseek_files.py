import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

ROOT_PATH = config.root_path

def load_json(filename):
    """读取 JSON 文件内容并返回数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 构造 repo_list_1870-1003.json 的路径
    repo_list_file = os.path.join(ROOT_PATH, 'repo_list_1870-1003.json')

    try:
        repos = load_json(repo_list_file)
    except Exception as e:
        print(f"读取文件 {repo_list_file} 时出错：{e}")
        return

    count_removed = 0

    for repo in repos:
        # 获取每个代码库的名称
        repo_name = repo.get('name')
        if not repo_name:
            continue
        # 构造目标文件的完整路径：config.root_path/result/<repo name>/is_opt_keyword_deepseek.json
        target_file = os.path.join(ROOT_PATH, 'result', repo_name, 'is_opt_keyword_deepseek.json')
        
        if os.path.exists(target_file):
            try:
                os.remove(target_file)
                count_removed += 1
                print(f"已删除文件: {target_file}")
            except Exception as e:
                print(f"删除文件 {target_file} 失败：{e}")
        else:
            print(f"文件不存在: {target_file}")
    
    print(f"\n删除完成，共删除了 {count_removed} 个文件。")

if __name__ == "__main__":
    main()
