import json
import os
import sys
from collections import defaultdict, Counter

# 添加 python/ 目录到 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def check_duplicate_names(input_file):
    """
    分析代码库名称的统计信息，包括：
    1. 检查 name 字段的重复情况
    2. 检查 name_long 字段的唯一性
    3. 分析所有者（Owner）信息
    4. 检查特殊字符
    5. 输出总结统计
    """
    print(f"Reading from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        repos = json.load(f)
    
    print(f"Total repositories: {len(repos)}")
    print("=" * 100)
    
    # 1. 检查 name 的重复情况
    print(f"\n1. Name duplication check:")
    name_counter = Counter(repo.get('name') for repo in repos)
    name_duplicates = {name: count for name, count in name_counter.items() if count > 1}
    
    print(f"   - Unique names: {len(name_counter)}")
    print(f"   - Duplicate names: {len(name_duplicates)}")
    
    if name_duplicates:
        print(f"   - Duplicates detail:")
        # 使用字典记录每个 name 对应的所有 name_long
        name_to_name_longs = defaultdict(list)
        
        for repo in repos:
            name = repo.get("name")
            if name in name_duplicates:
                name_long = repo.get("name_long")
                http_url = repo.get("http_url")
                
                name_to_name_longs[name].append({
                    "name_long": name_long,
                    "http_url": http_url
                })
        
        # 按重复次数排序
        sorted_duplicates = sorted(name_to_name_longs.items(), key=lambda x: len(x[1]), reverse=True)
        
        for name, info_list in sorted_duplicates:
            print(f"\n     Name: '{name}' (appears {len(info_list)} times)")
            print(f"     {'-' * 76}")
            for i, info in enumerate(info_list, 1):
                print(f"       {i}. name_long: {info['name_long']}")
                print(f"          http_url:  {info['http_url']}")
    
    # 2. 检查 name_long 的唯一性
    print(f"\n2. Name_long uniqueness check:")
    name_long_counter = Counter(repo.get('name_long') for repo in repos)
    name_long_duplicates = {nl: count for nl, count in name_long_counter.items() if count > 1}
    
    print(f"   - Unique name_long: {len(name_long_counter)}")
    print(f"   - Duplicate name_long: {len(name_long_duplicates)}")
    if name_long_duplicates:
        print(f"   - Duplicates detail:")
        for nl, count in sorted(name_long_duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"     * '{nl}' appears {count} times")
    
    # 3. 检查不同的 name 对应同一个 owner 的情况
    owner_to_repos = defaultdict(list)
    for repo in repos:
        name_long = repo.get('name_long', '')
        if '_' in name_long:
            parts = name_long.rsplit('_', 1)
            if len(parts) == 2:
                owner = parts[0]
                repo_name = parts[1]
                owner_to_repos[owner].append({
                    'name': repo.get('name'),
                    'name_long': name_long,
                    'http_url': repo.get('http_url')
                })
    
    print(f"\n3. Owner analysis:")
    print(f"   - Total unique owners: {len(owner_to_repos)}")
    
    # 找出拥有多个代码库的 owner
    multi_repo_owners = {owner: repos_list for owner, repos_list in owner_to_repos.items() if len(repos_list) > 1}
    print(f"   - Owners with multiple repositories: {len(multi_repo_owners)}")
    
    if multi_repo_owners:
        print(f"   - Top 10 owners by repository count:")
        sorted_owners = sorted(multi_repo_owners.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        for owner, repos_list in sorted_owners:
            print(f"     * {owner}: {len(repos_list)} repositories")
            for r in repos_list[:3]:  # 只显示前3个
                print(f"       - {r['name']} ({r['name_long']})")
            if len(repos_list) > 3:
                print(f"       ... and {len(repos_list) - 3} more")
    
    # 4. 检查是否有 name 字段包含下划线或斜杠的情况
    name_with_special = [repo for repo in repos if '_' in repo.get('name', '') or '/' in repo.get('name', '')]
    
    print(f"\n4. Special characters in name:")
    print(f"   - Names with '_' or '/': {len(name_with_special)}")
    if name_with_special[:5]:
        print(f"   - Examples:")
        for repo in name_with_special[:5]:
            print(f"     * name: {repo.get('name')} | name_long: {repo.get('name_long')}")
    
    # 5. 统计信息
    print(f"\n5. Summary:")
    print(f"   - Total repositories: {len(repos)}")
    print(f"   - Unique 'name' values: {len(name_counter)}")
    print(f"   - Unique 'name_long' values: {len(name_long_counter)}")
    print(f"   - All 'name' are unique: {len(name_counter) == len(repos)}")
    print(f"   - All 'name_long' are unique: {len(name_long_counter) == len(repos)}")
    if name_duplicates:
        total_repos_with_duplicates = sum(name_duplicates.values())
        print(f"   - Repositories with duplicate names: {total_repos_with_duplicates}")
        print(f"   - Percentage: {total_repos_with_duplicates / len(repos) * 100:.2f}%")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    # 输入文件路径
    input_file = os.path.join(config.root_path, "repo_list_30342_with_name_long.json")
    
    print(f"Input file: {input_file}")
    print()
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        exit(1)
    
    check_duplicate_names(input_file)

