import json
import re
import os
import sys

# 添加 python/ 目录到 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def extract_full_name_from_url(http_url):
    """
    从 GitHub URL 中提取 用户名/仓库名 格式
    例如：https://github.com/Genymobile/scrcpy.git -> Genymobile/scrcpy
    """
    # 匹配 https://github.com/用户名/仓库名.git
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', http_url)
    if match:
        owner = match.group(1)
        repo = match.group(2)
        return f"{owner}/{repo}"
    return None

def create_name_long(full_name):
    """
    将 用户名/仓库名 格式转换为 用户名_仓库名 格式
    例如：Genymobile/scrcpy -> Genymobile_scrcpy
    """
    if full_name:
        return full_name.replace("/", "_")
    return None

def add_name_long_field(input_file, output_file):
    """
    读取输入JSON文件，为每个代码库添加 name_long 字段，并输出到新文件
    """
    print(f"Reading from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        repos = json.load(f)
    
    print(f"Total repositories: {len(repos)}")
    
    # 处理每个代码库
    success_count = 0
    fail_count = 0
    
    for repo in repos:
        # 从 http_url 提取完整名称
        http_url = repo.get("http_url", "")
        full_name = extract_full_name_from_url(http_url)
        
        if full_name:
            name_long = create_name_long(full_name)
            # 在 name 字段后面插入 name_long 字段
            # 创建新的有序字典
            new_repo = {}
            for key, value in repo.items():
                new_repo[key] = value
                if key == "name":
                    new_repo["name_long"] = name_long
            
            # 更新原字典
            repo.clear()
            repo.update(new_repo)
            success_count += 1
        else:
            print(f"Warning: Failed to extract full name from URL: {http_url}")
            # 即使失败也添加一个空的 name_long 字段
            new_repo = {}
            for key, value in repo.items():
                new_repo[key] = value
                if key == "name":
                    new_repo["name_long"] = None
            repo.clear()
            repo.update(new_repo)
            fail_count += 1
    
    print(f"Successfully processed: {success_count}")
    if fail_count > 0:
        print(f"Failed to process: {fail_count}")
    
    # 输出到新文件
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(repos, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Output saved to {output_file}")
    
    # 显示一些示例
    print("\nSample entries (first 3):")
    for i, repo in enumerate(repos[:3]):
        print(f"\nRepo {i+1}:")
        print(f"  name: {repo.get('name')}")
        print(f"  name_long: {repo.get('name_long')}")
        print(f"  http_url: {repo.get('http_url')}")

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = os.path.join(config.root_path, "repo_list_30342.json")
    output_file = os.path.join(config.root_path, "repo_list_30342_with_name_long.json")
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    add_name_long_field(input_file, output_file)

