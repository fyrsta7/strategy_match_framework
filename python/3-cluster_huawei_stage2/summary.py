import os
# 强制使用CPU，避免GPU显存不足
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import sys
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")
INPUT_FILE = "summary_huawei.json"
OUTPUT_FILE = "summary_huawei.json"

# 复用来源配置
REUSE_SOURCES = [
    {
        'summary_file': 'summary.json',
        'field_name': 'optimization_summary_final',
        'label': 'general'  # 用于统计标记
    },
    {
        'summary_file': 'summary_arch.json',
        'field_name': 'optimization_summary_arch_final',
        'label': 'arch'
    }
]

# LLM 配置
MODEL_NAME = config.xmcp_deepseek_model
NUM_SUMMARIES = 3  # Self-consistency voting 数量
MAX_REPO_WORKERS = 16  # 代码库级别并行数
MAX_WORKERS = 8  # commit 级别并行数
SKIP_PROCESSED = True  # 是否跳过已处理的 commit
MAX_LLM_RETRIES = 3  # LLM 最大重试次数
TEMPERATURE_MIN = 0  # 温度下限
TEMPERATURE_MAX = 0.5  # 温度上限

# 初始化句子转换模型用于计算相似度
model_path = os.path.join(config.root_path, "models/all-MiniLM-L6-v2")
sentence_model = SentenceTransformer(model_path, device='cpu')

# 初始化OpenAI客户端
client = OpenAI(
    base_url=config.xmcp_base_url,
    api_key=config.xmcp_api_key_unlimit,
)

# 全局统计和锁
global_stats_lock = threading.Lock()
global_stats = {
    'total_commits': 0,
    'reused_from_general': 0,
    'reused_from_arch': 0,
    'generated': 0,
    'skipped': 0,
    'failed': 0
}

# 全局进度条（用于commit级别）
global_commit_pbar = None

# System prompt
system_prompt = (
    "You are an expert code optimization analyst specializing in identifying and summarizing performance optimization patterns from git commits. "
    "Your expertise includes analyzing code changes in C/C++ programs to extract generalizable optimization strategies. "
    "Your summaries help developers identify similar optimization opportunities in their own codebases by providing clear, one-sentence descriptions that balance specificity with generalizability."
)

def call_llm(prompt):
    """通用的 llm 调用函数，返回回复内容，使用随机生成的temperature值"""
    temperature = random.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"调用LLM出错: {e}")
        time.sleep(3)
        return None

def get_diff_by_commit(repo_name, commit_hash):
    """从预存储的diff.txt文件中获取代码差异"""
    try:
        diff_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, "modified_file", commit_hash, "diff.txt")
        
        if not os.path.exists(diff_file_path):
            return None
        
        with open(diff_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
            return content
            
    except Exception as e:
        print(f"读取diff文件出错 [{repo_name}:{commit_hash}]: {str(e)}")
        return None

def calculate_similarity(text1, text2):
    """计算两段文本的相似度"""
    embedding1 = sentence_model.encode([text1])[0]
    embedding2 = sentence_model.encode([text2])[0]
    
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

def build_reuse_index(repositories):
    """
    构建复用索引：{(repo_name, commit_hash): (summary, source_label)}
    返回: (index, stats)
    """
    print("\n构建复用索引...")
    index = {}
    stats = {
        'total_repos_checked': 0,
        'general_found': 0,
        'arch_found': 0,
        'total_indexed': 0
    }
    
    for repo_name in tqdm(repositories, desc="构建索引", unit="repo"):
        stats['total_repos_checked'] += 1
        
        for source_config in REUSE_SOURCES:
            file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, source_config['summary_file'])
            
            if not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    commits = json.load(f)
                
                for commit in commits:
                    # 使用 repository_name 或当前 repo_name
                    commit_repo = commit.get('repository_name', repo_name)
                    commit_hash = commit.get('hash')
                    
                    if not commit_hash:
                        continue
                    
                    key = (commit_repo, commit_hash)
                    
                    # 优先使用第一个找到的 summary
                    if key not in index:
                        summary = commit.get(source_config['field_name'])
                        if summary and summary.strip():
                            index[key] = (summary, source_config['label'])
                            stats['total_indexed'] += 1
                            
                            if source_config['label'] == 'general':
                                stats['general_found'] += 1
                            elif source_config['label'] == 'arch':
                                stats['arch_found'] += 1
            
            except Exception as e:
                pass  # 忽略读取错误
    
    print(f"索引构建完成：")
    print(f"  检查的代码库数: {stats['total_repos_checked']}")
    print(f"  从 summary.json 索引: {stats['general_found']} 个")
    print(f"  从 summary_arch.json 索引: {stats['arch_found']} 个")
    print(f"  总索引数: {stats['total_indexed']} 个")
    
    return index, stats

def process_commit(commit, repo_name, reuse_index):
    """处理单个commit：优先复用，否则生成新的summary"""
    # 检查是否需要跳过已处理的
    if SKIP_PROCESSED and "optimization_summary_huawei_final" in commit:
        if commit.get("optimization_summary_huawei_final"):
            return {
                "optimization_summary_huawei": commit.get("optimization_summary_huawei", []),
                "optimization_summary_huawei_final": commit["optimization_summary_huawei_final"],
                "reused_from": commit.get("reused_from", "existing"),
                "status": "skipped"
            }
    
    commit_hash = commit.get('hash', '')
    commit_repo = commit.get('repository_name', repo_name)
    key = (commit_repo, commit_hash)
    
    # 尝试复用
    if key in reuse_index:
        summary, source = reuse_index[key]
        return {
            "optimization_summary_huawei": [summary] * NUM_SUMMARIES,
            "optimization_summary_huawei_final": summary,
            "reused_from": source,
            "status": "reused"
        }
    
    # 生成新 summary
    commit_message = commit.get('message', '')
    modified_files = ", ".join(commit.get('modified_files', []))
    modified_func = ", ".join(commit.get('modified_func', []))
    
    diff_content = get_diff_by_commit(commit_repo, commit_hash)
    
    if diff_content is None:
        return {
            "status": "no_diff",
            "reason": "diff_file_not_found"
        }
    
    # 构建 prompt
    base_prompt = (
        "Please analyze the following commit that implements performance optimization. "
        "Your task is to provide a ONE-SENTENCE summary in English describing an optimization strategy that can be applied to similar code patterns. "
        "The summary must be exactly one sentence - no more, no less.\n\n"
        
        "IMPORTANT: If the optimization matches one of the common patterns below, use the provided template summary. "
        "Otherwise, create a summary following the same style.\n\n"
        
        "COMMON OPTIMIZATION PATTERNS AND THEIR SUMMARIES:\n"
        "1. Changing 'container.size() == 0' to 'container.empty()' → "
        "Replace size() == 0 checks with empty() method calls for better performance and readability.\n"
        "2. Reordering conditions in if statements with multiple 'and'-connected sub-conditions → "
        "Reorder sub-conditions in if statements with multiple 'and'-connected conditions to place simpler or less expensive checks before more complex ones for better short-circuit evaluation.\n"
        "3. Caching repeated calculations → "
        "Cache frequently computed values to avoid redundant calculations in loops or repeated calls.\n\n"
        
        "GUIDELINES for creating optimization summaries:\n"
        "1. Include the main code structure or pattern where the optimization applies (e.g., 'in for loops', 'when using containers', 'in recursive functions')\n"
        "2. Describe the specific change made within that code context\n"
        "3. Include relevant technical details that are important for applying the optimization\n"
        "4. Make the description specific enough to identify similar code patterns but general enough to apply across different projects\n\n"
        
        "EXAMPLE of including code context:\n"
        "- If a commit changes from pass-by-value to pass-by-reference in for loop iteration: "
        "Use pass-by-reference instead of pass-by-value when iterating over containers in for loops to avoid unnecessary object copying.\n"
        "- If a commit optimizes string concatenation in loops: "
        "Use string builders or pre-allocated buffers instead of repeated string concatenation operations within loops to reduce memory allocations.\n"
        "- If a commit changes vector push_back to reserve: "
        "Pre-allocate container capacity using reserve() before adding multiple elements to avoid repeated memory reallocations during insertion.\n\n"
        
        "This commit has already been confirmed as a performance optimization. "
        "Analyze the code changes and create a summary that includes both the code context and the specific optimization technique.\n\n"
        
        f"Repository: {commit_repo}\n"
        f"Commit Hash: {commit_hash}\n"
        f"Commit Message: {commit_message}\n"
        f"Modified Files: {modified_files}\n"
        f"Modified Functions: {modified_func}\n\n"
        f"Code Changes:\n{diff_content}\n\n"
        
        "Please provide your response as exactly one sentence that describes both the code context (where to apply) and the optimization strategy (what to change)."
    )
    
    summaries = []
    for i in range(NUM_SUMMARIES):
        prompt = f"Query ID: {i+1} for commit {commit_hash}\n\n" + base_prompt
        
        retry_count = 0
        while retry_count < MAX_LLM_RETRIES:
            response = call_llm(prompt)
            if response is not None and response.strip():
                summaries.append(response.strip())
                break
            retry_count += 1
            time.sleep(1)
    
    if not summaries:
        return {
            "status": "failed",
            "reason": "llm_failed"
        }
    
    # Self-consistency voting
    if len(summaries) == 1:
        final_summary = summaries[0]
    else:
        similarity_matrix = np.zeros((len(summaries), len(summaries)))
        for i in range(len(summaries)):
            for j in range(len(summaries)):
                if i != j:
                    similarity_matrix[i][j] = calculate_similarity(summaries[i], summaries[j])
        
        avg_similarities = np.mean(similarity_matrix, axis=1)
        best_summary_index = np.argmax(avg_similarities)
        final_summary = summaries[best_summary_index]
    
    return {
        "optimization_summary_huawei": summaries,
        "optimization_summary_huawei_final": final_summary,
        "reused_from": "generated",
        "status": "generated"
    }

def process_repository(repo_name, reuse_index):
    """处理单个代码库"""
    global global_commit_pbar
    
    json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, INPUT_FILE)
    
    if not os.path.exists(json_file_path):
        return {
            "repo_name": repo_name,
            "status": "file_not_found",
            "stats": {"total": 0, "reused": 0, "generated": 0, "skipped": 0, "failed": 0}
        }
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
    except Exception as e:
        return {
            "repo_name": repo_name,
            "status": "read_error",
            "error": str(e),
            "stats": {"total": 0, "reused": 0, "generated": 0, "skipped": 0, "failed": 0}
        }
    
    if not commits:
        return {
            "repo_name": repo_name,
            "status": "empty_file",
            "stats": {"total": 0, "reused": 0, "generated": 0, "skipped": 0, "failed": 0}
        }
    
    stats = {
        "total": len(commits),
        "reused": 0,
        "generated": 0,
        "skipped": 0,
        "failed": 0,
        "reused_general": 0,
        "reused_arch": 0
    }
    
    # 处理每个 commit
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {}
        for i, commit in enumerate(commits):
            future = executor.submit(process_commit, commit, repo_name, reuse_index)
            future_to_index[future] = i
        
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                result = future.result()
                
                if result['status'] == 'reused':
                    commits[i]['optimization_summary_huawei'] = result['optimization_summary_huawei']
                    commits[i]['optimization_summary_huawei_final'] = result['optimization_summary_huawei_final']
                    commits[i]['reused_from'] = result['reused_from']
                    stats['reused'] += 1
                    
                    if result['reused_from'] == 'general':
                        stats['reused_general'] += 1
                    elif result['reused_from'] == 'arch':
                        stats['reused_arch'] += 1
                
                elif result['status'] == 'generated':
                    commits[i]['optimization_summary_huawei'] = result['optimization_summary_huawei']
                    commits[i]['optimization_summary_huawei_final'] = result['optimization_summary_huawei_final']
                    commits[i]['reused_from'] = result['reused_from']
                    stats['generated'] += 1
                
                elif result['status'] == 'skipped':
                    stats['skipped'] += 1
                
                else:
                    stats['failed'] += 1
                
                # 更新全局commit进度条
                if global_commit_pbar is not None:
                    with global_stats_lock:
                        global_commit_pbar.update(1)
                        global_commit_pbar.set_postfix({
                            "复用": global_stats['reused_from_general'] + global_stats['reused_from_arch'],
                            "生成": global_stats['generated'],
                            "跳过": global_stats['skipped']
                        })
            
            except Exception as e:
                print(f"\n[异常] 代码库 {repo_name}, commit {i}: {str(e)}")
                stats['failed'] += 1
                
                # 即使出错也要更新进度条
                if global_commit_pbar is not None:
                    with global_stats_lock:
                        global_commit_pbar.update(1)
    
    # 保存结果
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(commits, f, ensure_ascii=False, indent=2)
        
        return {
            "repo_name": repo_name,
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        return {
            "repo_name": repo_name,
            "status": "write_error",
            "error": str(e),
            "stats": stats
        }

def main():
    """主函数"""
    global global_commit_pbar
    
    print("=" * 80)
    print("生成优化策略总结（带复用机制）")
    print("=" * 80)
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"知识库路径: {KNOWLEDGE_BASE_PATH}")
    print(f"输入/输出文件: {INPUT_FILE}")
    print(f"LLM 模型: {MODEL_NAME}")
    print(f"每个 commit 生成 {NUM_SUMMARIES} 个总结进行投票")
    print(f"代码库级别并行数: {MAX_REPO_WORKERS}")
    print(f"Commit级别并行数: {MAX_WORKERS}")
    print("-" * 80)
    
    # 读取代码库列表
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件不存在 - {REPO_LIST_FILE}")
        return
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repo_list = json.load(f)
    
    repositories = []
    for repo in repo_list:
        repo_name = repo.get('name_long') or repo.get('name')
        if repo_name:
            repositories.append(repo_name)
    
    if not repositories:
        print("错误：未找到任何代码库")
        return
    
    # 只处理存在 summary_huawei.json 的代码库
    valid_repos = []
    for repo_name in repositories:
        json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, INPUT_FILE)
        if os.path.exists(json_file_path):
            valid_repos.append(repo_name)
    
    print(f"发现 {len(valid_repos)} 个包含 {INPUT_FILE} 的代码库")
    
    if not valid_repos:
        print("错误：没有需要处理的代码库")
        return
    
    # 构建复用索引
    reuse_index, index_stats = build_reuse_index(repositories)
    
    # 统计总的commit数量
    print("\n统计总commit数量...")
    total_commits_count = 0
    for repo_name in tqdm(valid_repos, desc="统计commits", unit="repo"):
        json_file_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name, INPUT_FILE)
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                commits = json.load(f)
                total_commits_count += len(commits)
        except:
            pass
    
    print(f"总共需要处理 {total_commits_count} 个 commits")
    
    # 处理所有代码库
    print(f"\n开始处理 {len(valid_repos)} 个代码库...")
    start_time = time.time()
    
    results = []
    
    # 创建两个进度条：一个用于代码库，一个用于commit
    repo_pbar = tqdm(total=len(valid_repos), desc="代码库进度", position=0, unit="repo")
    global_commit_pbar = tqdm(total=total_commits_count, desc="Commit进度", position=1, unit="commit")
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_REPO_WORKERS) as executor:
            future_to_repo = {executor.submit(process_repository, repo, reuse_index): repo for repo in valid_repos}
            
            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    with global_stats_lock:
                        if result["status"] == "success":
                            global_stats['total_commits'] += result["stats"]["total"]
                            global_stats['reused_from_general'] += result["stats"].get("reused_general", 0)
                            global_stats['reused_from_arch'] += result["stats"].get("reused_arch", 0)
                            global_stats['generated'] += result["stats"]["generated"]
                            global_stats['skipped'] += result["stats"]["skipped"]
                            global_stats['failed'] += result["stats"]["failed"]
                    
                    # 更新代码库进度条
                    repo_pbar.update(1)
                    repo_pbar.set_postfix({
                        "已处理repos": len(results),
                        "成功": sum(1 for r in results if r["status"] == "success")
                    })
                
                except Exception as e:
                    print(f"\n[异常] 处理代码库 {repo_name}: {str(e)}")
                    repo_pbar.update(1)
    
    finally:
        # 确保进度条关闭
        repo_pbar.close()
        if global_commit_pbar is not None:
            global_commit_pbar.close()
    
    total_time = time.time() - start_time
    
    # 输出最终统计
    print("\n" + "=" * 80)
    print("复用统计报告")
    print("=" * 80)
    print(f"总 commit 数: {global_stats['total_commits']}")
    print(f"从 summary.json 复用: {global_stats['reused_from_general']} ({(global_stats['reused_from_general']/global_stats['total_commits']*100):.1f}%)")
    print(f"从 summary_arch.json 复用: {global_stats['reused_from_arch']} ({(global_stats['reused_from_arch']/global_stats['total_commits']*100):.1f}%)")
    print(f"新生成: {global_stats['generated']} ({(global_stats['generated']/global_stats['total_commits']*100):.1f}%)")
    print(f"跳过（已存在）: {global_stats['skipped']}")
    print(f"失败: {global_stats['failed']}")
    
    total_reused = global_stats['reused_from_general'] + global_stats['reused_from_arch']
    print(f"\n总复用率: {(total_reused/global_stats['total_commits']*100):.1f}%")
    
    saved_llm_calls = total_reused * NUM_SUMMARIES
    actual_llm_calls = global_stats['generated'] * NUM_SUMMARIES
    print(f"\n节省 LLM 调用次数: {total_reused} × {NUM_SUMMARIES} = {saved_llm_calls} 次")
    print(f"实际 LLM 调用次数: {global_stats['generated']} × {NUM_SUMMARIES} = {actual_llm_calls} 次")
    
    print(f"\n总处理时间: {total_time:.1f}s")
    print("\n处理完成！")

if __name__ == "__main__":
    main()

