import requests
import os
import json
from datetime import datetime, timedelta
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 搜索参数
LANGUAGES = ["C", "C++"]
STAR_THRESHOLD = 100
STAR_UPPER_BOUND = 200000  # 假设一个较高上界，超过此值的仓库基本不可能
DAYS_THRESHOLD = 5 * 365
MAX_REPOS_PER_QUERY = 1000  # 单个查询区间最多返回的结果（API 限制）
MAX_REPO_SIZE_GB = 3         # 最大允许的代码库大小 (GB)
MAX_REPO_SIZE_KB = MAX_REPO_SIZE_GB * 1024 * 1024  # 转换为KB单位
PER_PAGE = 100  # GitHub API 允许的最大每页数量
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_new.json")

def wait_for_rate_limit(response):
    """根据响应头中的 X-RateLimit-Reset 计算需要等待的秒数，并sleep。"""
    reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
    sleep_duration = max(reset_time - int(time.time()) + 1, 0)
    print(f"Rate limit reached. Sleeping for {sleep_duration} seconds...")
    time.sleep(sleep_duration)

def search_repositories_interval(language, star_lower, star_upper):
    """
    根据语言及给定的 star 数区间查询仓库。
    如果返回结果数达到 API 限制（1000），则对子区间递归查询。
    """
    date_threshold = (datetime.now() - timedelta(days=DAYS_THRESHOLD)).strftime("%Y-%m-%d")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"token {config.GITHUB_TOKEN}"
    query = f"language:{language} stars:{star_lower}..{star_upper} pushed:>{date_threshold}"
    
    # 先请求一页获取总结果数
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": 1,
        "page": 1
    }
    response = requests.get(
        f"{config.GITHUB_API_URL}/search/repositories",
        headers=headers,
        params=params
    )
    if response.status_code == 403:
        wait_for_rate_limit(response)
        # 重新发起请求
        response = requests.get(
            f"{config.GITHUB_API_URL}/search/repositories",
            headers=headers,
            params=params
        )
    if response.status_code != 200:
        print(f"Error fetching {language} stars {star_lower}..{star_upper}: {response.status_code}")
        print(f"Response: {response.text}")
        return []
    data = response.json()
    total_count = data.get("total_count", 0)
    
    # 如果返回数达到 API 限制（1000），则需要对子区间再细分
    if total_count >= MAX_REPOS_PER_QUERY and star_lower < star_upper:
        mid = (star_lower + star_upper) // 2
        print(f"Splitting interval {star_lower}..{star_upper} into {star_lower}..{mid} and {mid+1}..{star_upper} (total_count~{total_count})")
        left = search_repositories_interval(language, star_lower, mid)
        right = search_repositories_interval(language, mid+1, star_upper)
        return left + right

    # 否则，正常分页获取所有结果
    collected_repos = []
    page = 1
    while True:
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": PER_PAGE,
            "page": page
        }
        response = requests.get(
            f"{config.GITHUB_API_URL}/search/repositories",
            headers=headers,
            params=params
        )
        if response.status_code == 403:
            wait_for_rate_limit(response)
            continue  # 重新请求当前页面
        if response.status_code != 200:
            print(f"Error fetching {language} page {page} for stars {star_lower}..{star_upper}: {response.status_code}")
            print(f"Response: {response.text}")
            break
        data = response.json()
        items = data.get("items", [])
        # 过滤仓库大小
        filtered = [
            repo for repo in items 
            if repo.get("size", 0) <= MAX_REPO_SIZE_KB
        ]
        collected_repos.extend(filtered)
        if len(items) < PER_PAGE:
            break
        page += 1
    print(f"Collected {len(collected_repos)} repos for {language} with stars in range {star_lower}..{star_upper} (approx total_count: {total_count})")
    return collected_repos

def load_existing_repos(json_path):
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

if __name__ == "__main__":
    os.makedirs(os.path.dirname(REPO_LIST_FILE), exist_ok=True)
    existing_repos = load_existing_repos(REPO_LIST_FILE)
    existing_urls = {repo["http_url"] for repo in existing_repos}
    all_repos = []
    for lang in LANGUAGES:
        print(f"\n{'='*30}\nSearching {lang} repositories with stars >= {STAR_THRESHOLD} ...")
        repos = search_repositories_interval(lang, STAR_THRESHOLD, STAR_UPPER_BOUND)
        print(f"Found {len(repos)} {lang} repositories in total")
        all_repos.extend(repos)
    # 过滤已存在的仓库
    new_repos = [
        {
            "name": repo["name"],
            "http_url": repo["clone_url"],
            "ssh_url": repo["ssh_url"]
        }
        for repo in all_repos
        if repo["clone_url"] not in existing_urls
    ]
    if new_repos:
        existing_repos.extend(new_repos)
        with open(REPO_LIST_FILE, "w") as f:
            json.dump(existing_repos, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully added {len(new_repos)} new repositories")
        print(f"Total repositories now: {len(existing_repos)}")
    else:
        print("\nNo new repositories found to add")
    print(f"\nSearch completed. Results saved to: {REPO_LIST_FILE}")
