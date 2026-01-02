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
REPO_PARALLEL_WORKERS = 8  # 仓库级别的并行数（同时处理多少个仓库）
MAX_WORKERS = 32  # 最大线程数（用于每个仓库内的commit并行处理）
SKIP_PROCESSED = False  # 是否跳过已经处理过的commit（设为True表示跳过）
USE_REFERENCE_RESULTS = False  # 是否使用现有的参考结果
REFERENCE_FILENAME = "reference_results.json"  # 参考结果文件名
KNOWLEDGE_BASE_ROOT = os.path.join(config.root_path, "knowledge_base")  # 知识库根目录

# 文件名配置
INPUT_KEYWORD_FILE = "is_opt_arch_keyword.json"  # 关键词筛选结果
LLM_PROCESSING_FILE = "is_opt_arch_llm.json"    # LLM处理中间文件
FINAL_OUTPUT_FILE = "is_opt_arch_final.json"    # 最终通过LLM的结果
ALL_REPOS_SUMMARY_FILE = "all_is_opt_arch_final.json"  # 汇总文件

# 文件复制配置
OVERWRITE_LLM_FILE = True  # 当is_opt_arch_llm.json已存在时，是否覆盖
COPY_PARALLEL_WORKERS = 256  # 文件复制阶段的并行数

# LLM结果字段名
LLM_RESULT_FIELD = "is_opt_arch_llm"  # LLM判断结果字段

LLM_MODEL = config.xmcp_deepseek_model
repo_list_file = os.path.join(config.root_path, "repo_list_30342.json")  # 设置代码库列表文件路径
SUMMARY_FILE_PATH = os.path.join(config.root_path, ALL_REPOS_SUMMARY_FILE)  # 汇总文件路径

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


def copy_single_repo_file(repo):
    """
    复制单个仓库的关键词筛选结果到LLM处理文件
    返回操作结果
    """
    repo_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo)
    source_file = os.path.join(repo_path, INPUT_KEYWORD_FILE)
    target_file = os.path.join(repo_path, LLM_PROCESSING_FILE)
    
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        return {"repo": repo, "status": "skipped", "reason": "源文件不存在"}
    
    # 检查目标文件是否已存在
    if os.path.exists(target_file):
        if not OVERWRITE_LLM_FILE:
            return {"repo": repo, "status": "skipped", "reason": "目标文件已存在且未开启覆盖"}
    
    try:
        # 复制文件
        shutil.copy2(source_file, target_file)
        return {"repo": repo, "status": "success"}
    except Exception as e:
        return {"repo": repo, "status": "error", "error": str(e)}


def copy_keyword_to_llm_files(repositories):
    """
    阶段1：复制关键词筛选结果到LLM处理文件
    支持并行处理多个仓库
    """
    print("=" * 60)
    print("阶段1：复制关键词筛选结果")
    print("=" * 60)
    print(f"📊 待处理代码库数量: {len(repositories)}")
    print(f"⚡ 最大并行数: {COPY_PARALLEL_WORKERS}")
    print(f"📝 覆盖模式: {'开启' if OVERWRITE_LLM_FILE else '关闭'}")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=COPY_PARALLEL_WORKERS) as executor:
        futures = []
        for repo in repositories:
            future = executor.submit(copy_single_repo_file, repo)
            futures.append((future, repo))
        
        # 收集结果并显示进度
        success_count = 0
        skipped_count = 0
        error_count = 0
        
        for future, repo in tqdm(futures, desc="📋 复制文件", unit="repo"):
            try:
                result = future.result()
                if result["status"] == "success":
                    success_count += 1
                elif result["status"] == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1
                    print(f"  ❌ {repo}: {result.get('error', '未知错误')}")
            except Exception as e:
                print(f"  ❌ 处理仓库 {repo} 时发生异常: {e}")
                error_count += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("📊 复制统计:")
    print(f"  ✅ 成功复制: {success_count}")
    print(f"  ⏭️  跳过: {skipped_count}")
    print(f"  ❌ 失败: {error_count}")
    print("=" * 60)

def get_system_prompt():
    """根据配置动态生成system prompt"""
    base_prompt = """You are an expert in computer architecture, low-level performance optimization, and Git commit analysis. Your task is to determine whether a given Git commit meets ALL of the following criteria:

1. The primary purpose of the commit is performance optimization (specifically reducing runtime resource consumption such as execution time or memory usage)
2. The optimization is directly related to computer architecture or hardware features, such as:
   - CPU architecture-specific optimizations (x86, ARM, RISC-V, etc.)
   - SIMD and vectorization (SSE, AVX, NEON, etc.)
   - Cache optimizations (cache alignment, prefetching, reducing cache misses)
   - Memory architecture (NUMA, memory alignment, memory ordering, memory barriers)
   - Atomic operations and synchronization primitives
   - CPU features (intrinsics, inline assembly, branch prediction, pipelining)
3. The optimization technique used is relatively generic and transferable to other functions or codebases that have similar architecture-related performance characteristics
4. The changes do not primarily focus on code readability, maintainability, or other non-performance-related improvements

The optimization must be architecture-related and have the MAIN goal of improving performance. Generic algorithmic improvements, data structure changes, or high-level optimizations that do not leverage architecture-specific features should be excluded.

You must answer strictly with "true" or "false":
- Answer "true" ONLY if the commit meets ALL four criteria above
- Answer "false" if the commit does not meet any of the criteria, or if the optimization is not architecture-related, or if it is too specific/context-dependent

Do not provide any explanation, reasoning, or additional text in your response. Only return "true" or "false"."""
    
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

def get_user_prompt():
    """根据配置动态生成user prompt"""
    prompt_parts = ["Here is the information for a Git commit:"]
    
    if USE_COMMIT_MESSAGE:
        prompt_parts.append("\nCommit Message:\n{}")
    
    if USE_DIFF_CONTENT:
        prompt_parts.append("\nGit Diff:\n{}")
    
    if USE_BEFORE_FUNC:
        prompt_parts.append("\nFunction before change (complete):\n{}")
    
    prompt_parts.append("""

This commit only modifies one function in one file. Based on the information provided above, does this commit meet all the following criteria:
1. Primary purpose is performance optimization (reducing runtime resource consumption)
2. The optimization is directly related to computer architecture or hardware features (e.g., SIMD, cache optimization, memory alignment, atomic operations, CPU-specific instructions, etc.)
3. Uses a relatively generic optimization technique that could be transferred to other functions/codebases with similar architecture-related performance characteristics
4. Not primarily focused on readability or maintainability improvements

Answer "true" only if ALL criteria are met, otherwise answer "false". 

Note: Generic algorithmic improvements or data structure changes that do not leverage architecture-specific features should be answered as "false".""")
    
    return "".join(prompt_parts).strip()

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
            return {commit["hash"]: commit.get(LLM_RESULT_FIELD, "unknown") for commit in reference_data}
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

def update_commit_result(file_path, commit_hash, result):
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
                    commit[LLM_RESULT_FIELD] = result
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

def query_llm(commit_message, diff_content, before_func_content):
    """
    调用 LLM 进行筛选，返回 "true" 或 "false"，如果调用失败则返回 "unknown"。
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
            return "unknown"
        
        user_prompt_template = get_user_prompt()
        formatted_prompt = user_prompt_template.format(*prompt_args)
        
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": formatted_prompt},
        ]
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages
        )
        
        result = response.choices[0].message.content.strip().lower()
        return result if result in ["true", "false"] else "unknown"
    except Exception as e:
        print(f"[LLM] 查询失败: {e}")
        return "unknown"

def filter_commits_from_json_by_llm_parallel(repo_name, file_path, max_workers):
    """
    利用 LLM 对 file_path 文件中 commit 进行筛选，更新其中 is_opt_arch_llm 字段。
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
        if SKIP_PROCESSED and commit.get(LLM_RESULT_FIELD, "unknown") != "unknown":
            continue
            
        # 如果启用了参考结果，检查参考结果中是否有对应的结果
        if USE_REFERENCE_RESULTS and commit_hash in reference_results:
            ref_result = reference_results[commit_hash]
            if ref_result in ["true", "false"]:
                # 直接更新文件中的结果
                update_commit_result(file_path, commit_hash, ref_result)
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
                query_llm, 
                commit_message,
                diff_content,
                before_func_content
            )
            futures.append((future, commit_hash))
        
        # 处理结果并实时更新文件
        for future, commit_hash in tqdm(futures, desc="LLM filtering", unit="commit"):
            try:
                result = future.result()
                # 实时更新文件中的结果
                success = update_commit_result(file_path, commit_hash, result)
                if not success:
                    print(f"[LLM] 更新 commit {commit_hash} 结果失败")
            except Exception as e:
                print(f"[LLM] 处理 commit {commit_hash} 失败: {e}")
                # 即使LLM调用失败，也尝试更新为unknown
                update_commit_result(file_path, commit_hash, "unknown")
    
    print(f"[LLM] LLM 筛选完成，结果已实时保存在 {file_path}.")

def process_single_repo(repo, result_root, commit_workers):
    """
    处理单个仓库的LLM筛选任务
    """
    result_path = os.path.join(result_root, repo)
    target_file = os.path.join(result_path, LLM_PROCESSING_FILE)
    
    if not os.path.exists(target_file):
        print(f"[LLM] 仓库 {repo}：没有找到 {target_file}，跳过。")
        return
    
    # 验证文件格式
    test_data = safe_read_json(target_file)
    if not test_data:
        print(f"[LLM] 仓库 {repo}：文件 {target_file} 无法读取或为空，跳过。")
        return
    
    # 确保所有commit都有is_opt_arch_llm字段
    updated = False
    for commit in test_data:
        if LLM_RESULT_FIELD not in commit:
            commit[LLM_RESULT_FIELD] = "unknown"
            updated = True
    
    if updated:
        success = safe_write_json(target_file, test_data)
        if success:
            print(f"[LLM] 仓库 {repo}：已为缺失的commit添加 {LLM_RESULT_FIELD} 字段。")
        else:
            print(f"[LLM] 仓库 {repo}：更新字段失败，跳过。")
            return
            
    # 对文件进行LLM筛选
    filter_commits_from_json_by_llm_parallel(repo, target_file, commit_workers)

def extract_single_repo_final(repo):
    """
    从单个仓库的LLM处理结果中提取通过的commits
    返回操作结果
    """
    repo_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo)
    source_file = os.path.join(repo_path, LLM_PROCESSING_FILE)
    target_file = os.path.join(repo_path, FINAL_OUTPUT_FILE)
    
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        return {"repo": repo, "status": "skipped", "reason": "源文件不存在"}
    
    try:
        # 读取LLM处理结果
        with open(source_file, "r", encoding="utf-8") as f:
            all_commits = json.load(f)
        
        # 筛选通过LLM判断的commits
        passed_commits = [
            commit for commit in all_commits
            if commit.get(LLM_RESULT_FIELD) == "true"
        ]
        
        # 写入最终结果文件
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(passed_commits, f, indent=4, ensure_ascii=False)
        
        return {
            "repo": repo,
            "status": "success",
            "total": len(all_commits),
            "passed": len(passed_commits)
        }
    except Exception as e:
        return {"repo": repo, "status": "error", "error": str(e)}


def extract_final_results(repositories):
    """
    阶段3：提取所有通过LLM筛选的commits
    支持并行处理多个仓库
    """
    print("\n" + "=" * 60)
    print("阶段3：提取通过LLM筛选的commits")
    print("=" * 60)
    print(f"📊 待处理代码库数量: {len(repositories)}")
    print(f"⚡ 最大并行数: {COPY_PARALLEL_WORKERS}")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=COPY_PARALLEL_WORKERS) as executor:
        futures = []
        for repo in repositories:
            future = executor.submit(extract_single_repo_final, repo)
            futures.append((future, repo))
        
        # 收集结果并显示进度
        success_count = 0
        skipped_count = 0
        error_count = 0
        total_commits = 0
        passed_commits = 0
        
        for future, repo in tqdm(futures, desc="📤 提取最终结果", unit="repo"):
            try:
                result = future.result()
                if result["status"] == "success":
                    success_count += 1
                    total_commits += result.get("total", 0)
                    passed_commits += result.get("passed", 0)
                elif result["status"] == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1
                    print(f"  ❌ {repo}: {result.get('error', '未知错误')}")
            except Exception as e:
                print(f"  ❌ 处理仓库 {repo} 时发生异常: {e}")
                error_count += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("📊 提取统计:")
    print(f"  ✅ 成功处理: {success_count}")
    print(f"  ⏭️  跳过: {skipped_count}")
    print(f"  ❌ 失败: {error_count}")
    print(f"  📝 总commit数: {total_commits}")
    print(f"  ✨ 通过LLM筛选: {passed_commits}")
    if total_commits > 0:
        print(f"  📈 通过率: {passed_commits / total_commits * 100:.2f}%")
    print("=" * 60)


def process_llm_phase(repositories, result_root, max_workers, repo_parallel_workers):
    """
    阶段2：针对所有代码库，对 is_opt_arch_llm.json 文件中未处理的 commit 利用 LLM 并行筛选，
    更新字段 is_opt_arch_llm。
    
    支持两层并行：
    1. 仓库级别并行：同时处理多个仓库
    2. Commit级别并行：每个仓库内的commits并行处理
    """
    print("\n" + "=" * 60)
    print("阶段2：LLM筛选")
    print("=" * 60)
    
    # 打印当前配置
    config_info = []
    if USE_COMMIT_MESSAGE:
        config_info.append("Commit Message")
    if USE_DIFF_CONTENT:
        config_info.append("Diff Content")
    if USE_BEFORE_FUNC:
        config_info.append("Before Function")
    
    print(f"[LLM] 当前信息源配置: {', '.join(config_info) if config_info else '无'}")
    print(f"[LLM] 仓库级别并行数: {repo_parallel_workers}")
    print(f"[LLM] 每个仓库内commit并行数: {max_workers}")
    
    # 计算每个仓库实际可用的线程数（平均分配）
    commit_workers_per_repo = max(1, max_workers // repo_parallel_workers)
    print(f"[LLM] 每个仓库分配的线程数: {commit_workers_per_repo}")
    
    # 使用线程池并行处理多个仓库
    with ThreadPoolExecutor(max_workers=repo_parallel_workers) as repo_executor:
        futures = []
        for repo in repositories:
            future = repo_executor.submit(
                process_single_repo,
                repo,
                result_root,
                commit_workers_per_repo
            )
            futures.append((future, repo))
        
        # 等待所有仓库处理完成
        for future, repo in tqdm(futures, desc="🤖 LLM筛选仓库", unit="repo"):
            try:
                future.result()
            except Exception as e:
                print(f"[LLM] 处理仓库 {repo} 时发生错误: {e}")

def aggregate_all_results(repositories):
    """
    阶段4：汇总所有仓库的最终结果到根目录
    """
    print("\n" + "=" * 60)
    print("阶段4：汇总所有仓库结果")
    print("=" * 60)
    print(f"📊 待汇总代码库数量: {len(repositories)}")
    
    all_commits = []
    repo_stats = {}
    success_repos = 0
    skipped_repos = 0
    
    for repo in tqdm(repositories, desc="📥 汇总结果", unit="repo"):
        repo_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo)
        final_file = os.path.join(repo_path, FINAL_OUTPUT_FILE)
        
        # 检查文件是否存在
        if not os.path.exists(final_file):
            skipped_repos += 1
            continue
        
        try:
            # 读取最终结果
            with open(final_file, "r", encoding="utf-8") as f:
                repo_commits = json.load(f)
            
            # 为每个commit添加repository字段
            for commit in repo_commits:
                commit["repository"] = repo
            
            all_commits.extend(repo_commits)
            repo_stats[repo] = len(repo_commits)
            success_repos += 1
            
        except Exception as e:
            print(f"  ❌ 读取仓库 {repo} 的结果失败: {e}")
            skipped_repos += 1
    
    # 将汇总结果写入文件
    try:
        with open(SUMMARY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(all_commits, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 汇总结果已保存到: {SUMMARY_FILE_PATH}")
    except Exception as e:
        print(f"\n❌ 保存汇总结果失败: {e}")
        return
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("📊 汇总统计:")
    print(f"  ✅ 成功读取仓库: {success_repos}")
    print(f"  ⏭️  跳过仓库: {skipped_repos}")
    print(f"  📝 总commit数: {len(all_commits)}")
    
    # 打印Top 10仓库的commit数量
    if repo_stats:
        print("\n📈 Top 10 仓库（按commit数量）:")
        sorted_repos = sorted(repo_stats.items(), key=lambda x: x[1], reverse=True)
        for i, (repo, count) in enumerate(sorted_repos[:10], 1):
            print(f"  {i}. {repo}: {count} commits")
    
    print("=" * 60)


if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_BASE_ROOT):
        print(f"Error: 目录 '{KNOWLEDGE_BASE_ROOT}' 不存在。")
        sys.exit(1)
    
    # 从 JSON 文件中读取代码库列表
    try:
        with open(repo_list_file, 'r', encoding='utf-8') as f:
            repo_list_data = json.load(f)
        
        # 根据 repo_list_30342.json 的结构提取代码库名称
        if isinstance(repo_list_data, list):
            repositories = []
            for repo_info in repo_list_data:
                if isinstance(repo_info, dict) and 'name' in repo_info:
                    repositories.append(repo_info['name'])
                elif isinstance(repo_info, str):
                    # 兼容性：如果元素是字符串，直接使用
                    repositories.append(repo_info)
            
            # 过滤掉空字符串
            repositories = [name for name in repositories if name]
        else:
            raise ValueError("代码库列表文件格式错误：期望数组格式")
        
        print(f"从 {repo_list_file} 读取到 {len(repositories)} 个代码库")
        print(f"代码库列表: {', '.join(repositories[:5])}{'...' if len(repositories) > 5 else ''}")
        
    except FileNotFoundError:
        print(f"错误：找不到代码库列表文件 {repo_list_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误：解析 JSON 文件失败 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取代码库列表时发生错误 - {e}")
        sys.exit(1)
    
    if not repositories:
        print("未找到任何代码仓库")
        sys.exit(1)
    
    # 阶段1：复制关键词筛选结果
    copy_keyword_to_llm_files(repositories)
    
    # 阶段2：LLM筛选（两层并行：仓库级别并行 + 每个代码库内commit级别并行）
    process_llm_phase(repositories, KNOWLEDGE_BASE_ROOT, MAX_WORKERS, REPO_PARALLEL_WORKERS)
    
    # 阶段3：提取通过LLM筛选的commits
    extract_final_results(repositories)
    
    # 阶段4：汇总所有仓库结果
    aggregate_all_results(repositories)
    
    print("\n" + "=" * 60)
    print("✅ 所有处理阶段完成！")
    print("=" * 60)