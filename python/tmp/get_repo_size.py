import json
import re
import time
import requests
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

CONFIG_PATH = config.root_path

# 从 HTTP URL 中解析 owner 和 repository 名称的正则表达式，
# 示例：https://github.com/zhaohengbo/rt-uboot.git
REPO_URL_REGEX = re.compile(r'https://github\.com/([^/]+)/([^/]+)(\.git)?')

# GitHub REST API 模板
GITHUB_API_REPO_URL = "https://api.github.com/repos/{owner}/{repo}"

# GitHub API 请求超时，等待时间（秒）
RETRY_DELAY = 10

def get_repo_api_url(http_url):
    """
    根据 http_url 提取 owner 和 repo，并生成 GitHub API URL
    """
    m = REPO_URL_REGEX.match(http_url)
    if not m:
        raise ValueError(f"无法解析仓库 URL: {http_url}")
    owner, repo = m.group(1), m.group(2)
    return GITHUB_API_REPO_URL.format(owner=owner, repo=repo)

def get_repo_size_in_gb(api_url):
    """
    调用 GitHub API 获取仓库信息，返回仓库大小（GB）
    如果调用超时，则等待一段时间后重试
    """
    while True:
        try:
            response = requests.get(api_url, timeout=10)
            # 如果因速率限制或其他原因非 200，可以短暂等待后重试
            if response.status_code != 200:
                print(f"请求 {api_url} 返回状态码 {response.status_code}，等待 {RETRY_DELAY} 秒后重试…")
                time.sleep(RETRY_DELAY)
                continue

            data = response.json()
            # GitHub API 返回的 size 字段单位为 KB
            size_kb = data.get("size")
            if size_kb is None:
                raise ValueError("返回数据中没有 size 字段")
            size_gb = size_kb / (1024 * 1024)  # 转换为 GB
            return size_gb
        except requests.exceptions.Timeout:
            print(f"请求 {api_url} 超时，等待 {RETRY_DELAY} 秒后重试…")
            time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            # 其他网络相关的异常也等待后重试
            print(f"请求 {api_url} 出现异常: {e}，等待 {RETRY_DELAY} 秒后重试…")
            time.sleep(RETRY_DELAY)

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"配置文件 {CONFIG_PATH} 不存在")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        repos = json.load(f)

    # 遍历所有仓库，调用 GitHub API 获取大小信息，并加入新的 key
    for repo in repos:
        http_url = repo.get("http_url")
        if not http_url:
            print(f"仓库 {repo.get('name')} 没有 http_url 信息，跳过")
            continue

        try:
            api_url = get_repo_api_url(http_url)
        except ValueError as e:
            print(e)
            continue

        print(f"正在获取 {repo.get('name')} 的信息：{api_url}")
        size_gb = get_repo_size_in_gb(api_url)
        repo["size_gb"] = size_gb
        print(f"{repo.get('name')} 大小为：{size_gb:.4f} GB")
    
    # 保存更新后的数据到配置文件，注意这里直接覆盖原文件
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(repos, f, indent=4)
    
    print("所有仓库信息已更新并写回到配置文件。")

if __name__ == '__main__':
    main()