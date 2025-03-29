import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

ROOT_PATH = config.root_path

def load_json(filename):
    """加载 JSON 文件内容"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {filename} 失败：{e}")
        return None

def main():
    # 构造 repo_list_1870-1003.json 文件路径
    repo_list_file = os.path.join(ROOT_PATH, 'repo_list_1870-1003.json')
    
    repos = load_json(repo_list_file)
    if repos is None:
        return
    
    # 提取所有代码库名称（确保 name 字段存在）
    included_repositories = [repo.get('name') for repo in repos if repo.get('name')]
    
    # 整理为 Python 列表形式，并添加变量名 INCLUDED_REPOSITORIES
    output_str = "INCLUDED_REPOSITORIES = " + json.dumps(included_repositories, indent=4, ensure_ascii=False)
    
    # 将结果写入到一个 Python 文件中（如 included_repositories.py），以便其他地方直接 import 使用
    output_file = os.path.join(ROOT_PATH, 'included_repositories.py')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 该文件由 generate_included_repositories.py 自动生成，列出了所有代码库名称\n")
            f.write(output_str)
            f.write("\n")
        print(f"已生成文件 {output_file}")
    except Exception as e:
        print(f"写入文件 {output_file} 失败：{e}")

if __name__ == "__main__":
    main()
