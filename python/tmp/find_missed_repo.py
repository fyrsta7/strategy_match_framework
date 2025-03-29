import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def find_missing_repositories(root_path):
    """
    判断哪些代码库在 repo_list.json 中存在，但在 repository 文件夹中不存在。
    """
    # 设置 repo_list.json 路径
    json_path = os.path.join(root_path, "repo_list_30345_2.json")
    # 设置 repository 文件夹路径
    repo_folder_path = os.path.join(root_path, "repository")

    # 读取 repo_list.json 文件
    try:
        with open(json_path, "r") as f:
            repo_list = json.load(f)
    except FileNotFoundError:
        print(f"错误：{json_path} 文件未找到！请检查路径。")
        return
    except json.JSONDecodeError:
        print(f"错误：{json_path} 文件的 JSON 格式不正确！")
        return

    # 提取 repo_list.json 中的项目名称
    repo_names = {repo["name"] for repo in repo_list}

    # 获取 repository 文件夹中的子文件夹名称
    if not os.path.exists(repo_folder_path):
        print(f"错误：{repo_folder_path} 文件夹未找到！请检查路径。")
        return
    repo_folders = {name for name in os.listdir(repo_folder_path) if os.path.isdir(os.path.join(repo_folder_path, name))}

    # 找出 repo_list.json 中存在但 repository 文件夹中不存在的项目
    missing_repos = repo_names - repo_folders

    # 输出结果
    if missing_repos:
        print(f"在 {json_path} 中存在，但在 {repo_folder_path} 中缺失的代码库有 {len(missing_repos)} 个：")
        for repo in missing_repos:
            print(repo)
    else:
        print("所有代码库都已存在于 repository 文件夹中。")


def count_repository_folders(root_path):
    """
    计算 repository 文件夹中有多少个子文件夹。
    """
    repo_folder_path = os.path.join(root_path, "repository")
    
    # 检查 repository 文件夹是否存在
    if not os.path.exists(repo_folder_path):
        print(f"错误：{repo_folder_path} 文件夹未找到！请检查路径。")
        return 0

    # 统计子文件夹数量
    folder_count = sum(1 for name in os.listdir(repo_folder_path) if os.path.isdir(os.path.join(repo_folder_path, name)))
    print(f"repository 文件夹中共有 {folder_count} 个子文件夹。")
    return folder_count


def check_duplicate_repos(root_path):
    """
    检查 repo_list.json 中是否存在重复的代码库名称。
    """
    json_path = os.path.join(root_path, "repo_list_30345_2.json")

    # 读取 repo_list.json 文件
    try:
        with open(json_path, "r") as f:
            repo_list = json.load(f)
    except FileNotFoundError:
        print(f"错误：{json_path} 文件未找到！请检查路径。")
        return False
    except json.JSONDecodeError:
        print(f"错误：{json_path} 文件的 JSON 格式不正确！")
        return False

    # 提取所有的代码库名称
    repo_names = [repo["name"] for repo in repo_list]

    # 检查是否有重复项
    duplicates = {name for name in repo_names if repo_names.count(name) > 1}

    if duplicates:
        print(f"在 {json_path} 中发现重复的代码库名称：")
        for name in duplicates:
            print(name)
        return True
    else:
        print(f"{json_path} 中没有重复的代码库名称。")
        return False


if __name__ == "__main__":
    # 设置根路径（替换为你的实际路径）
    root_path = config.root_path

    # 执行函数
    print("检查缺失的代码库：")
    find_missing_repositories(root_path)

    print("\n统计 repository 文件夹中的子文件夹数量：")
    count_repository_folders(root_path)

    print("\n检查 repo_list.json 中是否存在重复的代码库：")
    check_duplicate_repos(root_path)