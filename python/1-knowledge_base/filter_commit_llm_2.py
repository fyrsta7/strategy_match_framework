import json
import os
import time
import shutil
import glob
import threading
from git import Repo
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局配置
MAX_WORKERS = 256  # 最大线程数
SKIP_PROCESSED = False  # 是否跳过已经处理过的commit（设为True表示跳过）
USE_REFERENCE_RESULTS = False  # 是否使用现有的参考结果
REFERENCE_FILENAME = "reference_results.json"  # 参考结果文件名
KNOWLEDGE_BASE_ROOT = os.path.join(config.root_path, "knowledge_base")  # 知识库根目录
FILE_NAME = "is_opt_llm_2.json"  # 输入输出文件名

# 信息使用配置
USE_COMMIT_MESSAGE = True   # 是否使用commit message信息
USE_DIFF_CONTENT = True    # 是否使用diff.txt信息
USE_BEFORE_FUNC = False     # 是否使用before_func信息

# 文件锁，用于防止并发写入冲突
file_locks = {}
lock_creation_lock = threading.Lock()

client = OpenAI(
    base_url=config.xmcp_base_url,
    api_key=config.xmcp_api_key_unlimit,
)

def get_file_lock(file_path):
    """获取文件对应的锁，如果不存在则创建"""
    with lock_creation_lock:
        if file_path not in file_locks:
            file_locks[file_path] = threading.Lock()
        return file_locks[file_path]

def get_system_prompt():
    """生成system prompt"""
    base_prompt = """You are an expert in software performance optimization and Git commit analysis. Your task is to analyze Git commits related to performance optimization.

Performance optimization specifically refers to modifications that can improve the runtime efficiency of the code or reduce the required resource overhead (such as execution time, memory usage, CPU usage, etc.).

A generic optimization approach refers to modifications that are relatively universal and do not require too much codebase-specific information, meaning they could potentially be migrated to other codebases to apply similar optimization patterns.

You must answer strictly with "true" or "false":
- Do not provide any explanation, reasoning, or additional text in your response
- Only return "true" or "false" as requested"""
    
    info_parts = []
    if USE_COMMIT_MESSAGE:
        info_parts.append("commit message")
    if USE_DIFF_CONTENT:
        info_parts.append("code diff")
    if USE_BEFORE_FUNC:
        info_parts.append("original function code")
    
    if info_parts:
        info_str = ", ".join(info_parts)
        base_prompt += f"\n\nYou will be provided with the following information: {info_str}."
    
    return base_prompt.strip()

def get_first_question_prompt():
    """生成第一个问题的user prompt"""
    prompt_parts = ["Here is the information for a Git commit:"]
    
    if USE_COMMIT_MESSAGE:
        prompt_parts.append("\nCommit Message:\n{}")
    
    if USE_DIFF_CONTENT:
        prompt_parts.append("\nGit Diff:\n{}")
    
    if USE_BEFORE_FUNC:
        prompt_parts.append("\nFunction before change (complete):\n{}")
    
    prompt_parts.append("""
This commit only modifies one function in one file. Based on the information provided above, is the primary purpose of this commit performance optimization (specifically to improve runtime efficiency or reduce resource overhead)?

Answer "true" if the main goal is performance optimization, otherwise answer "false".""")
    
    return "".join(prompt_parts).strip()

def get_second_question_prompt():
    """生成第二个问题的prompt"""
    return """Based on the same commit information, is the performance optimization approach used in this commit relatively generic? 

By "generic", I mean the optimization technique could potentially be applied to other functions or codebases without requiring too much specific knowledge about this particular codebase (e.g., algorithmic improvements, data structure optimizations, caching strategies, loop optimizations, etc.).

Answer "true" if the optimization approach is generic and transferable, otherwise answer "false"."""

def load_file_content(file_path):
    """
    加载文件内容，如果文件不存在或读取失败则返回空字符串
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def find_before_func_file(directory):
    """
    在指定目录中查找before_func.*文件，返回第一个匹配的文件路径
    如果没有找到，返回None
    """
    pattern = os.path.join(directory, "before_func.*")
    matching_files = glob.glob(pattern)
    if matching_files:
        return matching_files[0]  # 返回第一个匹配的文件
    return None

def load_reference_results(reference_file_path):
    """
    加载参考结果，返回一个哈希值到结果的映射
    """
    try:
        with open(reference_file_path, "r", encoding="utf-8") as f:
            reference_data = json.load(f)
            result_map = {}
            for commit in reference_data:
                commit_hash = commit["hash"]
                result_map[commit_hash] = {
                    "is_opt_ds_simple": commit.get("is_opt_ds_simple", "unknown"),
                    "is_general_ds_simple": commit.get("is_general_ds_simple", "unknown")
                }
            return result_map
    except Exception as e:
        print(f"[LLM] 加载参考结果文件 {reference_file_path} 失败: {e}")
        return {}

def safe_read_json(file_path):
    """线程安全地读取JSON文件"""
    file_lock = get_file_lock(file_path)
    with file_lock:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[LLM] 读取文件 {file_path} 失败: {e}")
            return []

def safe_write_json(file_path, data):
    """线程安全地写入JSON文件"""
    file_lock = get_file_lock(file_path)
    with file_lock:
        try:
            # 创建临时文件
            temp_file = file_path + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            
            # 原子性地替换原文件
            if os.path.exists(file_path):
                backup_file = file_path + ".backup"
                shutil.copy2(file_path, backup_file)
            
            shutil.move(temp_file, file_path)
            
            # 删除备份文件
            backup_file = file_path + ".backup"
            if os.path.exists(backup_file):
                os.remove(backup_file)
                
            return True
        except Exception as e:
            print(f"[LLM] 写入文件 {file_path} 失败: {e}")
            # 清理临时文件
            temp_file = file_path + ".tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

def update_commit_result(file_path, commit_hash, is_opt_result, is_general_result=None):
    """线程安全地更新单个commit的结果"""
    file_lock = get_file_lock(file_path)
    with file_lock:
        try:
            # 读取当前数据
            with open(file_path, "r", encoding="utf-8") as f:
                all_commits = json.load(f)
            
            # 更新指定commit的结果
            for commit in all_commits:
                if commit["hash"] == commit_hash:
                    commit["is_opt_ds_simple"] = is_opt_result
                    if is_general_result is not None:
                        commit["is_general_ds_simple"] = is_general_result
                    break
            
            # 写回文件
            temp_file = file_path + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(all_commits, f, indent=4)
            
            shutil.move(temp_file, file_path)
            return True
        except Exception as e:
            print(f"[LLM] 更新commit {commit_hash} 结果失败: {e}")
            # 清理临时文件
            temp_file = file_path + ".tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

def query_llm_two_stage(commit_message, diff_content, before_func_content):
    """
    分两阶段调用 LLM 进行筛选：
    1. 首先询问是否为性能优化
    2. 如果是性能优化，再询问是否为通用优化
    返回 (is_opt_result, is_general_result) 元组
    """
    try:
        # 根据配置决定使用哪些信息
        prompt_args = []
        if USE_COMMIT_MESSAGE:
            prompt_args.append(commit_message)
        if USE_DIFF_CONTENT:
            prompt_args.append(diff_content)
        if USE_BEFORE_FUNC:
            prompt_args.append(before_func_content)
        
        # 如果没有启用任何信息源，返回unknown
        if not prompt_args:
            print("[LLM] 警告：没有启用任何信息源")
            return "unknown", "unknown"
        
        # 第一阶段：询问是否为性能优化
        first_prompt_template = get_first_question_prompt()
        first_formatted_prompt = first_prompt_template.format(*prompt_args)
        
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": first_formatted_prompt},
        ]
        
        # 第一次LLM调用
        response1 = client.chat.completions.create(
            model=config.xmcp_deepseek_model,
            messages=messages
        )
        
        first_result = response1.choices[0].message.content.strip().lower()
        is_opt_result = first_result if first_result in ["true", "false"] else "unknown"
        
        # 如果第一阶段结果不是true，则不进行第二阶段
        if is_opt_result != "true":
            return is_opt_result, "unknown"
        
        # 第二阶段：询问是否为通用优化
        # 将第一轮的assistant回复添加到对话历史中
        messages.append({"role": "assistant", "content": response1.choices[0].message.content})
        messages.append({"role": "user", "content": get_second_question_prompt()})
        
        # 第二次LLM调用
        response2 = client.chat.completions.create(
            model=config.xmcp_deepseek_model,
            messages=messages
        )
        
        second_result = response2.choices[0].message.content.strip().lower()
        is_general_result = second_result if second_result in ["true", "false"] else "unknown"
        
        return is_opt_result, is_general_result
        
    except Exception as e:
        print(f"[LLM] 查询失败: {e}")
        return "unknown", "unknown"

def filter_commits_from_json_by_llm_parallel(repo_name, file_path, max_workers):
    """
    利用 LLM 对 file_path 文件中 commit 进行两阶段筛选，更新其中 is_opt_ds_simple 和 is_general_ds_simple 字段。
    根据SKIP_PROCESSED设置决定是否跳过已处理的commit。
    如果USE_REFERENCE_RESULTS为True，会尝试从参考文件获取结果。
    并行调用后，将更新后的 commit 列表写回 file_path。
    """
    all_commits = safe_read_json(file_path)
    if not all_commits:
        print(f"[LLM] 无法读取文件 {file_path} 或文件为空")
        return
    
    # 如果启用了参考结果，尝试加载参考文件
    reference_results = {}
    if USE_REFERENCE_RESULTS:
        reference_file_path = os.path.join(os.path.dirname(file_path), REFERENCE_FILENAME)
        if os.path.exists(reference_file_path):
            reference_results = load_reference_results(reference_file_path)
            print(f"[LLM] 已加载参考结果文件，包含 {len(reference_results)} 条记录")
    
    pending_commits = []
    updated_from_reference = 0
    
    for commit in all_commits:
        commit_hash = commit["hash"]
        
        # 先检查是否需要跳过已处理的commit
        if SKIP_PROCESSED:
            is_opt_done = commit.get("is_opt_ds_simple", "unknown") != "unknown"
            is_general_done = commit.get("is_general_ds_simple", "unknown") != "unknown"
            if is_opt_done and is_general_done:
                continue
            
        # 如果启用了参考结果，检查参考结果中是否有对应的结果
        if USE_REFERENCE_RESULTS and commit_hash in reference_results:
            ref_data = reference_results[commit_hash]
            ref_is_opt = ref_data.get("is_opt_ds_simple", "unknown")
            ref_is_general = ref_data.get("is_general_ds_simple", "unknown")
            
            if ref_is_opt in ["true", "false"]:
                # 直接更新文件中的结果
                update_commit_result(file_path, commit_hash, ref_is_opt, ref_is_general)
                updated_from_reference += 1
                continue
                
        pending_commits.append(commit)
    
    if updated_from_reference > 0:
        print(f"[LLM] 从参考结果中更新了 {updated_from_reference} 个commit的结果")
    
    print(f"[LLM] {file_path}：待处理 commit 数量：{len(pending_commits)}")
    if not pending_commits:
        print(f"[LLM] 文件 {file_path} 中所有 commit 均已有有效结果或被跳过，不进行 LLM 筛选。")
        return
    
    # 打印当前使用的信息源配置
    info_sources = []
    if USE_COMMIT_MESSAGE:
        info_sources.append("commit message")
    if USE_DIFF_CONTENT:
        info_sources.append("diff content")
    if USE_BEFORE_FUNC:
        info_sources.append("before function")
    print(f"[LLM] 使用信息源: {', '.join(info_sources)}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for commit in pending_commits:
            commit_hash = commit["hash"]
            modified_file_dir = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name, "modified_file", commit_hash)
            
            # 初始化内容变量
            commit_message = commit.get("message", "") if USE_COMMIT_MESSAGE else ""
            diff_content = ""
            before_func_content = ""
            
            # 根据配置获取对应的内容
            if USE_DIFF_CONTENT:
                diff_path = os.path.join(modified_file_dir, "diff.txt")
                diff_content = load_file_content(diff_path)
            
            if USE_BEFORE_FUNC:
                before_func_path = find_before_func_file(modified_file_dir)
                if before_func_path:
                    before_func_content = load_file_content(before_func_path)
                else:
                    print(f"[LLM] 警告：未找到 before_func 文件在目录 {modified_file_dir}")
            
            future = executor.submit(
                query_llm_two_stage, 
                commit_message,
                diff_content,
                before_func_content
            )
            futures.append((future, commit_hash))
        
        # 处理结果并实时更新文件
        for future, commit_hash in tqdm(futures, desc="LLM two-stage filtering", unit="commit"):
            try:
                is_opt_result, is_general_result = future.result()
                # 实时更新文件中的结果
                success = update_commit_result(file_path, commit_hash, is_opt_result, is_general_result)
                if not success:
                    print(f"[LLM] 更新 commit {commit_hash} 结果失败")
            except Exception as e:
                print(f"[LLM] 处理 commit {commit_hash} 失败: {e}")
                # 即使LLM调用失败，也尝试更新为unknown
                update_commit_result(file_path, commit_hash, "unknown", "unknown")
    
    print(f"[LLM] LLM 两阶段筛选完成，结果已实时保存在 {file_path}.")

def process_llm_phase(repositories, result_root, max_workers):
    """
    针对所有代码库，对 is_opt_llm.json 文件中未处理的 commit 利用 LLM 并行筛选，
    更新字段 is_opt_ds_simple 和 is_general_ds_simple。
    """
    print("\n===== LLM 两阶段筛选 =====")
    
    # 打印当前配置
    config_info = []
    if USE_COMMIT_MESSAGE:
        config_info.append("Commit Message")
    if USE_DIFF_CONTENT:
        config_info.append("Diff Content")
    if USE_BEFORE_FUNC:
        config_info.append("Before Function")
    
    print(f"[LLM] 当前信息源配置: {', '.join(config_info) if config_info else '无'}")
    print(f"[LLM] 两阶段筛选：1) 是否为性能优化 2) 是否为通用优化")
    
    for repo in tqdm(repositories, desc="LLM filtering per repository"):
        result_path = os.path.join(result_root, repo)
        target_file = os.path.join(result_path, FILE_NAME)
        
        if not os.path.exists(target_file):
            print(f"[LLM] 仓库 {repo}：没有找到 {target_file}，跳过。")
            continue
        
        # 验证文件格式
        test_data = safe_read_json(target_file)
        if not test_data:
            print(f"[LLM] 仓库 {repo}：文件 {target_file} 无法读取或为空，跳过。")
            continue
        
        # 确保所有commit都有相应字段
        updated = False
        for commit in test_data:
            if "is_opt_ds_simple" not in commit:
                commit["is_opt_ds_simple"] = "unknown"
                updated = True
            if "is_general_ds_simple" not in commit:
                commit["is_general_ds_simple"] = "unknown"
                updated = True
        
        if updated:
            success = safe_write_json(target_file, test_data)
            if success:
                print(f"[LLM] 仓库 {repo}：已为缺失的commit添加相应字段。")
            else:
                print(f"[LLM] 仓库 {repo}：更新字段失败，跳过。")
                continue
                
        # 对文件进行LLM两阶段筛选
        filter_commits_from_json_by_llm_parallel(repo, target_file, max_workers)

if __name__ == "__main__":
    repository_root = os.path.join(config.root_path, "repository")
    
    # 排除不处理的仓库
    EXCLUDED_REPOSITORIES = []
    
    if not os.path.exists(repository_root):
        print(f"Error: 目录 '{repository_root}' 不存在。")
        sys.exit(1)
        
    # repositories = [
    #     folder for folder in os.listdir(repository_root)
    #     if os.path.isdir(os.path.join(repository_root, folder)) and folder not in EXCLUDED_REPOSITORIES
    # ]
    repositories = ["rocksdb"]
    
    # LLM 两阶段筛选（并发执行，每个代码库内使用多线程处理）
    process_llm_phase(repositories, KNOWLEDGE_BASE_ROOT, MAX_WORKERS)
    print("\n所有仓库处理完成！")