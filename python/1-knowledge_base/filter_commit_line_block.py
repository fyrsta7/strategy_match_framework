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

# 全局变量：知识库根目录
KNOWLEDGE_BASE_ROOT = os.path.join(config.root_path, "knowledge_base_all")

# 全局变量，控制是否跳过已经处理过的代码库
SKIP_EXISTING_LINE_BLOCK_RESULTS = False

# 全局变量，控制代码行数和代码块数的上下限
MIN_TOTAL_CHANGED_LINES    = 1   # 修改代码行数下限
MAX_TOTAL_CHANGED_LINES    = 20  # 修改代码行数上限
MIN_MODIFIED_CODE_BLOCKS   = 1   # 修改代码块数下限  
MAX_MODIFIED_CODE_BLOCKS   = 5   # 修改代码块数上限

# 新增全局变量，控制是否使用行数和块数进行筛选
USE_LINE_FILTER  = True
USE_BLOCK_FILTER = True

# 并行处理的线程数
MAX_WORKERS = 128

# 全局变量，定义输入和输出的JSON文件名
INPUT_JSON_FILENAME  = "line_block.json"
OUTPUT_JSON_FILENAME = "func_name.json"

def process_single_repository(repo_name, knowledge_base_path):
    """
    处理单个代码库的函数，用于并行执行
    """
    input_file  = os.path.join(knowledge_base_path, INPUT_JSON_FILENAME)
    output_file = os.path.join(knowledge_base_path, OUTPUT_JSON_FILENAME)

    # 检查是否需要跳过
    if SKIP_EXISTING_LINE_BLOCK_RESULTS and os.path.exists(output_file):
        return (f"[LineBlock] 文件 '{output_file}' 已存在，跳过代码库 {repo_name}。", 0, 0)

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    result_msg, input_count, output_count = filter_commits_by_line_block(
        input_file, output_file, repo_name
    )
    return result_msg, input_count, output_count

def process_line_block_phase(repositories):
    """
    针对所有代码库，从 INPUT_JSON_FILENAME 读取 commit，
    根据行数和代码块数筛选，结果写到 OUTPUT_JSON_FILENAME。
    """
    print("===== 代码行数和代码块筛选阶段 =====")
    print(f"输入文件: {INPUT_JSON_FILENAME}")
    print(f"输出文件: {OUTPUT_JSON_FILENAME}")

    if USE_LINE_FILTER:
        print(f"行数筛选条件: [{MIN_TOTAL_CHANGED_LINES}, {MAX_TOTAL_CHANGED_LINES}]")
    if USE_BLOCK_FILTER:
        print(f"代码块数筛选条件: [{MIN_MODIFIED_CODE_BLOCKS}, {MAX_MODIFIED_CODE_BLOCKS}]")
    if not USE_LINE_FILTER and not USE_BLOCK_FILTER:
        print("警告: 未启用任何筛选条件，所有 commit 都将被保留")

    # 准备并行任务
    tasks = []
    for repo_name in repositories:
        kb_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name)
        tasks.append((repo_name, kb_path))

    results = []
    total_in  = 0
    total_out = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_repo = {
            executor.submit(process_single_repository, repo, kb): repo
            for repo, kb in tasks
        }

        for future in tqdm(as_completed(future_to_repo),
                           total=len(tasks),
                           desc="Line&Block filtering"):
            repo = future_to_repo[future]
            try:
                msg, cnt_in, cnt_out = future.result()
                results.append(msg)
                total_in  += cnt_in
                total_out += cnt_out
            except Exception as e:
                err = f"[LineBlock] 代码库 {repo} 处理时发生异常: {e}"
                results.append(err)
                print(err)

    # 打印结果
    for r in results:
        if r:
            print(r)

    print("\n===== 筛选统计 =====")
    print(f"输入文件总 commit 数: {total_in}")
    print(f"输出文件总 commit 数: {total_out}")
    if total_in > 0:
        rate = total_out / total_in * 100
        print(f"筛选通过率: {rate:.2f}%")
    print(f"输入文件名: {INPUT_JSON_FILENAME}")
    print(f"输出文件名: {OUTPUT_JSON_FILENAME}")

def filter_commits_by_line_block(input_file, output_file, repo_name):
    """
    从 input_file 读取 commit，按行数和块数筛选，写到 output_file。
    返回 (消息, 输入数量, 输出数量)。
    """
    try:
        if not os.path.exists(input_file):
            return f"[LineBlock] 错误：输入文件 '{input_file}' 不存在。", 0, 0

        with open(input_file, "r", encoding="utf-8") as f:
            all_commits = json.load(f)

        in_count = len(all_commits)
        filtered = []

        for c in all_commits:
            lines = c.get("total_changed_lines", 0)
            blocks = c.get("modified_code_blocks", 0)

            cond_line  = (not USE_LINE_FILTER)  or (MIN_TOTAL_CHANGED_LINES <= lines <= MAX_TOTAL_CHANGED_LINES)
            cond_block = (not USE_BLOCK_FILTER) or (MIN_MODIFIED_CODE_BLOCKS <= blocks <= MAX_MODIFIED_CODE_BLOCKS)

            if cond_line and cond_block:
                filtered.append(c)

        out_count = len(filtered)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=4)

        msg = (f"[LineBlock] 代码库 {repo_name}：从 {in_count} 个 commit 中 "
               f"找到 {out_count} 个符合条件的 commit，结果已保存到 {output_file}")
        return msg, in_count, out_count

    except Exception as e:
        return f"[LineBlock] 处理 {repo_name} 时发生错误: {e}", 0, 0

if __name__ == "__main__":
    # 使用全局变量 KNOWLEDGE_BASE_ROOT
    repository_root = KNOWLEDGE_BASE_ROOT

    # 可按需排除某些仓库
    EXCLUDED_REPOSITORIES = []

    if not os.path.exists(repository_root):
        print(f"Error: 目录 '{repository_root}' 不存在。")
        sys.exit(1)

    repositories = [
        d for d in os.listdir(repository_root)
        if os.path.isdir(os.path.join(repository_root, d))
           and d not in EXCLUDED_REPOSITORIES
    ]

    process_line_block_phase(repositories)
    print("\n所有仓库处理完成！")