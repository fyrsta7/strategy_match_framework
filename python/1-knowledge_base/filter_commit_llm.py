import json
import os
import time
import shutil
from git import Repo
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局配置
MAX_WORKERS = 512  # 最大线程数
SKIP_PROCESSED = True  # 是否跳过已经处理过的commit（设为True表示跳过）

client = OpenAI(
    base_url=config.xmcp_base_url,
    api_key=config.xmcp_api_key_unlimit,
)

system_prompt = """
You are an expert in software performance optimization and Git commit analysis. Your task is to determine whether a given Git commit primarily achieves performance optimization. Performance optimization refers specifically to reducing runtime resource consumption, such as decreasing execution time or memory usage, and does not include code readability, maintainability, or other non-performance-related changes.
You must answer strictly with "true" or "false":
- Answer "true" if the commit primarily achieves performance optimization.
- Answer "false" if the commit does not primarily achieve performance optimization.
Do not provide any explanation, reasoning, or additional text in your response. Only return "true" or "false".
""".strip()

user_prompt = """
Here is the information for a Git commit:

Commit Message:
{}

This commit only modifies one function in one file. Based on the information above, is this commit primarily achieving performance optimization?
""".strip()

'''
Git Diff:
{}

Function before change (complete):
{}
'''


def load_file_content(file_path):
    """
    加载文件内容，如果文件不存在或读取失败则返回空字符串
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def query_llm(commit_message, diff_content, before_func_content):
    """
    调用 LLM 进行筛选，返回 "true" 或 "false"，如果调用失败则返回 "unknown"。
    """
    try:
        formatted_prompt = user_prompt.format(
            # diff_content,
            # before_func_content,
            commit_message
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt},
        ]
        
        response = client.chat.completions.create(
            model=config.xmcp_deepseek_model,
            messages=messages
        )
        
        result = response.choices[0].message.content.strip().lower()
        return result if result in ["true", "false"] else "unknown"
    except Exception as e:
        print(f"[LLM] 查询失败: {e}")
        return "unknown"


def filter_commits_from_json_by_llm_parallel(repo_name, file_path, max_workers):
    """
    利用 LLM 对 file_path 文件中 commit 进行筛选，更新其中 is_opt_ds_simple 字段。
    根据SKIP_PROCESSED设置决定是否跳过已处理的commit。
    并行调用后，将更新后的 commit 列表写回 file_path。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            all_commits = json.load(f)
    except Exception as e:
        print(f"[LLM] 加载文件 {file_path} 失败: {e}")
        return

    pending_commits = []
    for commit in all_commits:
        if SKIP_PROCESSED and commit.get("is_opt_ds_simple", "unknown") != "unknown":
            continue
        pending_commits.append(commit)

    print(f"[LLM] {file_path}：待处理 commit 数量：{len(pending_commits)}")
    if not pending_commits:
        print(f"[LLM] 文件 {file_path} 中所有 commit 均已有有效结果或被跳过，不进行 LLM 筛选。")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for commit in pending_commits:
            commit_hash = commit["hash"]
            modified_file_dir = os.path.join(config.root_path, "knowledge_base", repo_name, "modified_file", commit_hash)
            
            # 获取diff内容和函数修改前内容
            diff_path = os.path.join(modified_file_dir, "diff.txt")
            before_func_path = os.path.join(modified_file_dir, "before_func.txt")
            
            diff_content = load_file_content(diff_path)
            before_func_content = load_file_content(before_func_path)
            
            future = executor.submit(
                query_llm, 
                commit["message"],
                diff_content,
                before_func_content
            )
            futures.append((future, commit))
        
        for future, commit_obj in tqdm(futures, desc="LLM filtering", unit="commit"):
            try:
                result = future.result()
                commit_obj["is_opt_ds_simple"] = result
            except Exception as e:
                print(f"[LLM] 处理 commit {commit_obj['hash']} 失败: {e}")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_commits, f, indent=4)
        print(f"[LLM] LLM 筛选完成，结果保存在 {file_path}.")
    except Exception as e:
        print(f"[LLM] 写入文件 {file_path} 出错: {e}")


def process_llm_phase(repositories, result_root, max_workers):
    """
    针对所有代码库，根据 one_func_deduplicate.json 生成（或更新） is_opt_llm.json 文件，
    然后对里面未处理的 commit 利用 LLM 并行筛选，更新字段 is_opt_ds_simple。
    """
    print("\n===== LLM 筛选 =====")
    for repo in tqdm(repositories, desc="LLM filtering per repository"):
        result_path = os.path.join(result_root, repo)
        input_file = os.path.join(result_path, "one_func_deduplicate.json")
        output_file = os.path.join(result_path, "is_opt_llm.json")
        
        if not os.path.exists(input_file):
            print(f"[LLM] 仓库 {repo}：没有找到 {input_file}，跳过。")
            continue
            
        # 加载输入文件
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                input_commits = json.load(f)
        except Exception as e:
            print(f"[LLM] 仓库 {repo}：读取输入文件 {input_file} 失败: {e}")
            continue
            
        # 检查输出文件是否存在，如果存在则更新内容
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    output_commits = json.load(f)
                
                # 创建输出文件中commit的hash映射
                output_commit_map = {commit["hash"]: commit for commit in output_commits}
                
                # 检查输入文件中是否有新的commit需要添加
                for input_commit in input_commits:
                    input_hash = input_commit["hash"]
                    if input_hash not in output_commit_map:
                        # 添加新的commit
                        input_commit["is_opt_ds_simple"] = "unknown"
                        output_commits.append(input_commit)
                
                # 更新输出文件
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_commits, f, indent=4)
                    
                print(f"[LLM] 仓库 {repo}：{output_file} 已更新，添加了新的commit。")
            except Exception as e:
                print(f"[LLM] 仓库 {repo}：更新输出文件失败: {e}")
                # 如果更新失败，使用输入文件的内容重新创建
                shutil.copy(input_file, output_file)
                print(f"[LLM] 仓库 {repo}：已重新创建 {output_file}")
        else:
            # 如果输出文件不存在，复制输入文件并添加is_opt_ds_simple字段
            try:
                for commit in input_commits:
                    commit["is_opt_ds_simple"] = "unknown"
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(input_commits, f, indent=4)
                    
                print(f"[LLM] 仓库 {repo}：{output_file} 不存在，已创建并初始化。")
            except Exception as e:
                print(f"[LLM] 仓库 {repo}：创建输出文件失败: {e}")
                continue
                
        # 对文件进行LLM筛选
        filter_commits_from_json_by_llm_parallel(repo, output_file, max_workers)


if __name__ == "__main__":
    repository_root = os.path.join(config.root_path, "repository")
    result_root = os.path.join(config.root_path, "knowledge_base")
    
    # 排除不处理的仓库
    EXCLUDED_REPOSITORIES = []
    
    if not os.path.exists(repository_root):
        print(f"Error: 目录 '{repository_root}' 不存在。")
        sys.exit(1)
        
    repositories = [
        folder for folder in os.listdir(repository_root)
        if os.path.isdir(os.path.join(repository_root, folder)) and folder not in EXCLUDED_REPOSITORIES
    ]
    
    # LLM 筛选（并发执行，每个代码库内使用多线程处理）
    process_llm_phase(repositories, result_root, MAX_WORKERS)
    print("\n所有仓库处理完成！")