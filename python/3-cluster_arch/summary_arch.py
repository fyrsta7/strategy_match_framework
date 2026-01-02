import os
# 强制使用CPU，避免GPU显存不足
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import sys
import time
import subprocess
import re
import random
import threading
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
from openai import OpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ============ 配置变量 ============
# 对每个commit生成的总结次数
NUM_GENERATIONS = 3
# LLM 的最大重试次数
MAX_LLM_RETRIES = 3
# 温度范围设置
TEMPERATURE_MIN = 0  # 温度下限
TEMPERATURE_MAX = 0.5  # 温度上限
# 如果commit已包含要生成的字段，是否重新生成
# False：使用已有结果并按需补足到 NUM_GENERATIONS 个
# True：强制重新生成所有结果
REGENERATE_EXISTING = False
# 仓库级别并行线程数
MAX_REPO_WORKERS = 32
# 单个仓库内commit并行线程数
MAX_WORKERS = 4
# 全局路径配置
KNOWLEDGE_BASE_ROOT = os.path.join(config.root_path, "knowledge_base")
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")
# 指定要处理的JSON文件名
JSON_FILE_NAME = "summary_arch.json"
# 使用的LLM模型
LLM_MODEL = config.xmcp_deepseek_model

# 初始化句子转换模型用于计算相似度（明确指定使用CPU）
model_path = os.path.join(config.root_path, "models/all-MiniLM-L6-v2")
sentence_model = SentenceTransformer(model_path, device='cpu')
# 初始化OpenAI客户端
client = OpenAI(
    base_url=config.xmcp_base_url,
    api_key=config.xmcp_api_key_unlimit,
)

# 全局统计和进度跟踪
global_stats_lock = threading.Lock()
global_progress_lock = threading.Lock()
global_no_diff_lock = threading.Lock()

system_prompt = (
    "You are an expert code optimization analyst specializing in computer architecture-specific performance optimizations in C/C++ programs. "
    "Your expertise includes analyzing code changes that leverage CPU architecture features, SIMD instructions, cache optimization, "
    "memory hierarchy, atomic operations, and other hardware-specific techniques. "
    "Your summaries help developers identify similar architecture-specific optimization opportunities in their own codebases by providing clear, "
    "one-sentence descriptions that balance specificity with generalizability."
)

def call_llm(prompt):
    """
    通用的 llm 调用函数，返回回复内容
    使用随机生成的temperature值
    """
    # 生成随机温度值
    temperature = random.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX)
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
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
        time.sleep(3)  # 发生错误时等待更长时间
        return None

def get_diff_by_commit(repo_name, commit_hash):
    """
    从预存储的diff.txt文件中获取代码差异
    """
    try:
        diff_file_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name, "modified_file", commit_hash, "diff.txt")
        
        if not os.path.exists(diff_file_path):
            return None  # 返回None表示文件不存在
        
        with open(diff_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None  # 返回None表示文件为空
            return content
            
    except Exception as e:
        print(f"读取diff文件出错 [{repo_name}:{commit_hash}]: {str(e)}")
        return None

def calculate_similarity(text1, text2):
    """
    计算两段文本的相似度
    """
    # 使用sentence-transformers编码文本
    embedding1 = sentence_model.encode([text1])[0]
    embedding2 = sentence_model.encode([text2])[0]
    
    # 计算余弦相似度
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

def process_commit_with_validation(commit, repo_name, expected_commit_id):
    """
    带验证的commit处理函数
    """
    # 验证commit_id匹配
    actual_commit_id = commit.get('__commit_id__', '')
    if actual_commit_id != expected_commit_id:
        raise ValueError(f"commit_id不匹配：期望{expected_commit_id}，实际{actual_commit_id}")
    
    # 调用原始处理函数
    result = process_commit(commit, repo_name)
    
    # 在结果中添加验证信息
    if result:
        result['__commit_id__'] = expected_commit_id
        result['__repo_name__'] = repo_name
        result['__original_index__'] = commit.get('__original_index__', -1)
    
    return result

def process_commit(commit, repo_name):
    """
    处理单个commit：基于Self-consistency voting生成多个优化策略总结，然后进行投票
    """
    # 检查commit是否已经有部分结果
    existing_summaries = commit.get("optimization_summary_arch", [])
    
    # 如果不重新生成且已有最终结果，则直接跳过
    if not REGENERATE_EXISTING and "optimization_summary_arch_final" in commit:
        return {
            "optimization_summary_arch": existing_summaries,
            "optimization_summary_arch_final": commit.get("optimization_summary_arch_final", ""),
            "skipped": True,
            "source": "existing"
        }
    
    commit_hash = commit.get('hash', '')
    commit_message = commit.get('message', '')
    modified_files = ", ".join(commit.get('modified_files', []))
    modified_func = ", ".join(commit.get('modified_func', []))
    
    # 获取代码差异
    diff_content = get_diff_by_commit(repo_name, commit_hash)
    
    # 如果无法获取diff内容，返回特殊标记表示需要跳过
    if diff_content is None:
        return {
            "skipped": True,
            "source": "no_diff",
            "reason": "diff_file_not_found"
        }
    
    # 构建优化后的提示（强调体系结构相关特性）
    base_prompt = (
        "Please analyze the following commit that implements a computer architecture-specific performance optimization. "
        "Your task is to provide a ONE-SENTENCE summary in English describing an architecture-specific optimization strategy that can be applied to similar code patterns. "
        "The summary must be exactly one sentence - no more, no less.\n\n"
        
        "IMPORTANT: This commit has been confirmed to be related to computer architecture features (CPU architecture, SIMD, cache, memory hierarchy, atomic operations, etc.). "
        "Your summary should clearly indicate the architecture-specific aspect of the optimization.\n\n"
        
        "COMMON ARCHITECTURE-SPECIFIC OPTIMIZATION PATTERNS:\n"
        "1. SIMD/Vectorization optimizations → "
        "Use SIMD instructions (SSE, AVX, NEON) to process multiple data elements in parallel for better throughput.\n"
        "2. Cache-friendly data layout → "
        "Reorganize data structures to improve cache locality and reduce cache misses by aligning data to cache line boundaries.\n"
        "3. Memory alignment optimizations → "
        "Align data structures to specific byte boundaries (16, 32, 64 bytes) to enable efficient SIMD operations and reduce memory access penalties.\n"
        "4. Atomic operation optimizations → "
        "Replace lock-based synchronization with lock-free atomic operations (CAS, fetch-and-add) to reduce contention on modern CPUs.\n"
        "5. Architecture-specific intrinsics → "
        "Use compiler intrinsics or inline assembly to leverage specific CPU features for better performance.\n"
        "6. Branch prediction hints → "
        "Add likely/unlikely hints or reorganize conditions to help CPU branch prediction and reduce pipeline stalls.\n"
        "7. Prefetching → "
        "Use software or hardware prefetching to bring data into cache before it's needed.\n\n"
        
        "GUIDELINES for creating architecture-specific optimization summaries:\n"
        "1. Identify the specific architecture feature being leveraged (SIMD, cache, memory, atomics, etc.)\n"
        "2. Describe the code pattern or context where the optimization applies\n"
        "3. Explain the specific change made to exploit the architecture feature\n"
        "4. Make the description specific enough to identify similar patterns but general enough to apply across different projects\n\n"
        
        "EXAMPLE of architecture-specific summaries:\n"
        "- SIMD optimization: Use AVX2 intrinsics to process 8 floating-point values simultaneously instead of scalar operations in tight loops.\n"
        "- Cache optimization: Reorganize struct members to place frequently accessed fields in the same cache line to improve cache locality.\n"
        "- Memory alignment: Align buffer allocations to 64-byte boundaries to match cache line size and enable efficient SIMD loads/stores.\n"
        "- Atomic optimization: Replace mutex-protected counter increments with atomic fetch-add operations for better scalability on multi-core systems.\n\n"
        
        "This commit has already been confirmed as an architecture-specific performance optimization. "
        "Analyze the code changes and create a summary that clearly highlights the architecture feature being exploited.\n\n"
        
        f"Repository: {repo_name}\n"
        f"Commit Hash: {commit_hash}\n"
        f"Commit Message: {commit_message}\n"
        f"Modified Files: {modified_files}\n"
        f"Modified Functions: {modified_func}\n\n"
        f"Code Changes:\n{diff_content}\n\n"
        
        "Please provide your response as exactly one sentence that describes both the architecture feature (what hardware aspect) "
        "and the optimization strategy (what code change) being applied."
    )
    
    # 用于存储多次尝试的结果
    summaries = existing_summaries.copy() if not REGENERATE_EXISTING else []
    
    # 生成需要的次数，如果已有部分结果且不需要重新生成，则只生成剩余需要的次数
    remaining_generations = NUM_GENERATIONS - len(summaries) if not REGENERATE_EXISTING else NUM_GENERATIONS
    
    # 生成指定次数的结果
    for i in range(remaining_generations):
        # 在prompt开头添加一个id，避免缓存命中
        prompt = f"Query ID: {i+1} for commit {commit_hash}\n\n" + base_prompt
        
        retry_count = 0
        while retry_count < MAX_LLM_RETRIES:
            response = call_llm(prompt)  # 每次调用都会使用随机temperature
            if response is not None and response.strip():
                summaries.append(response.strip())
                break
            retry_count += 1
            time.sleep(1)  # 短暂等待后重试
    
    # 如果没有生成任何有效结果，返回None
    if not summaries:
        return None
    
    # 通过Self-consistency voting选择最终结果
    # 对summary进行投票 - 选择最相似的群组
    if len(summaries) == 1:
        final_summary = summaries[0]
    else:
        similarity_matrix = np.zeros((len(summaries), len(summaries)))
        for i in range(len(summaries)):
            for j in range(len(summaries)):
                if i != j:
                    similarity_matrix[i][j] = calculate_similarity(summaries[i], summaries[j])
        
        # 计算每个summary的平均相似度
        avg_similarities = np.mean(similarity_matrix, axis=1)
        best_summary_index = np.argmax(avg_similarities)
        final_summary = summaries[best_summary_index]
    
    return {
        "optimization_summary_arch": summaries,
        "optimization_summary_arch_final": final_summary,
        "skipped": False,
        "source": "llm"
    }

def validate_processing_result(result, expected_commit_id):
    """
    验证处理结果的完整性
    """
    if not result:
        return False
    
    # 检查必要字段
    required_fields = ['optimization_summary_arch', 'optimization_summary_arch_final']
    for field in required_fields:
        if field not in result:
            return False
    
    # 检查commit_id匹配
    if '__commit_id__' in result and result.get('__commit_id__') != expected_commit_id:
        return False
    
    # 检查数据类型
    if not isinstance(result['optimization_summary_arch'], list):
        return False
    
    if not isinstance(result['optimization_summary_arch_final'], str):
        return False
    
    return True

def verify_result_belongs_to_commit(result, commit, commit_id):
    """
    最终验证：确保结果确实属于指定的commit
    """
    # 检查commit_id
    if '__commit_id__' in result and result.get('__commit_id__') != commit_id:
        return False
    
    # 检查原始索引（如果存在）
    if '__original_index__' in result and '__original_index__' in commit:
        if result.get('__original_index__') != commit.get('__original_index__'):
            return False
    
    return True

def safe_write_commits(json_file_path, updated_commits):
    """
    安全的文件写入，带备份和原子操作
    """
    # 1. 写入临时文件
    temp_file = f"{json_file_path}.tmp.{os.getpid()}.{int(time.time())}"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(updated_commits, f, ensure_ascii=False, indent=4)
        
        # 2. 验证临时文件
        with open(temp_file, 'r', encoding='utf-8') as f:
            verification_data = json.load(f)
        if len(verification_data) != len(updated_commits):
            raise ValueError("临时文件验证失败：数据长度不匹配")
        
        # 3. 创建备份（如果原文件存在）
        if os.path.exists(json_file_path):
            backup_file = f"{json_file_path}.backup.{int(time.time())}"
            shutil.copy2(json_file_path, backup_file)
        
        # 4. 原子性替换（在大多数文件系统上是原子操作）
        shutil.move(temp_file, json_file_path)
        
        return True
        
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        raise e

def count_total_commits(repositories):
    """
    预统计所有仓库的commit数量
    """
    repo_commit_counts = {}
    total_commits = 0
    
    for repo in repositories:
        repo_name = repo.get('name_long', repo.get('name', ''))
        if not repo_name:
            continue
            
        json_file_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name, JSON_FILE_NAME)
        
        if not os.path.exists(json_file_path):
            repo_commit_counts[repo_name] = 0
            continue
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                commits = json.load(f)
                commit_count = len(commits) if commits else 0
                repo_commit_counts[repo_name] = commit_count
                total_commits += commit_count
        except Exception:
            repo_commit_counts[repo_name] = 0
    
    return repo_commit_counts, total_commits

def update_global_progress(global_progress_bar, increment=1):
    """
    线程安全的全局进度更新函数
    """
    with global_progress_lock:
        global_progress_bar.update(increment)

def process_repository_safe(repo, repo_commit_counts, global_progress_bar, position_offset):
    """
    使用唯一标识符映射和线程安全的代码库处理函数
    """
    repo_name = repo.get('name_long', repo.get('name', ''))
    json_file_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name, JSON_FILE_NAME)
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"\n[错误] 代码库 {repo_name}: {JSON_FILE_NAME} 文件不存在 - {json_file_path}")
        return {
            "repo_name": repo_name,
            "status": "file_not_found",
            "stats": {"total": 0, "processed": 0, "skipped": 0, "failed": 0, "no_diff": 0}
        }
    
    try:
        # 读取commit数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
    except Exception as e:
        print(f"\n[错误] 代码库 {repo_name}: 读取 {JSON_FILE_NAME} 文件失败 - {str(e)}")
        return {
            "repo_name": repo_name,
            "status": "read_error",
            "error": str(e),
            "stats": {"total": 0, "processed": 0, "skipped": 0, "failed": 0, "no_diff": 0}
        }
    
    if not commits:
        print(f"\n[警告] 代码库 {repo_name}: {JSON_FILE_NAME} 文件为空或无有效commit数据")
        return {
            "repo_name": repo_name,
            "status": "empty_file",
            "stats": {"total": 0, "processed": 0, "skipped": 0, "failed": 0, "no_diff": 0}
        }
    
    # 线程安全的统计计数器和锁
    stats_lock = threading.Lock()
    results_lock = threading.Lock()
    stats = {"total": len(commits), "processed": 0, "skipped": 0, "failed": 0, "no_diff": 0}
    
    # 用于收集无diff文件的commit信息
    no_diff_commits = []
    no_diff_lock = threading.Lock()
    
    # 1. 创建严格的映射表
    commit_mapping = {}
    original_commits = []  # 保持原始顺序的commit列表
    
    for i, commit in enumerate(commits):
        # 生成更严格的唯一ID
        commit_hash = commit.get('hash', '')
        if not commit_hash:
            commit_hash = f'no_hash_{i}_{hash(str(commit))}'  # 使用内容hash作为fallback
        
        commit_id = f"{repo_name}::{commit_hash}::{i}"
        
        # 严格验证唯一性
        if commit_id in commit_mapping:
            raise ValueError(f"发现重复的commit_id: {commit_id}")
        
        # 创建处理用的commit副本（包含原始位置信息）
        commit_for_processing = commit.copy()
        commit_for_processing['__original_index__'] = i
        commit_for_processing['__commit_id__'] = commit_id
        
        commit_mapping[commit_id] = {
            'original_commit': commit,      # 原始commit对象（用于最终更新）
            'processing_commit': commit_for_processing,  # 处理用的副本
            'original_index': i,
            'processed': False              # 处理状态标记
        }
        
        original_commits.append(commit)
    
    # 2. 结果收集容器
    processing_results = {}
    failed_commit_ids = set()
    
    # 创建该仓库的commit进度条
    commit_pbar = tqdm(
        total=len(commits), 
        desc=f"[{repo_name}] commits", 
        position=position_offset, 
        leave=False,
        unit="commit"
    )
    
    # 3. 并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_commit_id = {}
        
        # 提交所有任务
        for commit_id, mapping_info in commit_mapping.items():
            future = executor.submit(
                process_commit_with_validation, 
                mapping_info['processing_commit'], 
                repo_name,
                commit_id  # 传递commit_id用于验证
            )
            future_to_commit_id[future] = commit_id
        
        # 收集结果
        for future in as_completed(future_to_commit_id):
            commit_id = future_to_commit_id[future]
            
            try:
                result = future.result()
                
                # 检查是否因为没有diff文件而跳过
                if result and result.get("reason") == "diff_file_not_found":
                    # 收集无diff文件的commit信息
                    original_commit = commit_mapping[commit_id]['original_commit']
                    commit_info = {
                        "repo": repo_name,
                        "hash": original_commit.get('hash', '未知'),
                        "message": original_commit.get('message', '未知')[:100] + "..." if len(original_commit.get('message', '')) > 100 else original_commit.get('message', '未知'),
                        "modified_files": original_commit.get('modified_files', [])
                    }
                    
                    with no_diff_lock:
                        no_diff_commits.append(commit_info)
                    
                    with stats_lock:
                        stats["no_diff"] += 1
                        status = "无diff文件"
                
                # 验证结果的完整性
                elif result and validate_processing_result(result, commit_id):
                    with results_lock:
                        processing_results[commit_id] = result
                        commit_mapping[commit_id]['processed'] = True
                    
                    with stats_lock:
                        if result.get("skipped", False):
                            stats["skipped"] += 1
                            status = "跳过"
                        else:
                            stats["processed"] += 1
                            status = "处理完成"
                else:
                    with results_lock:
                        failed_commit_ids.add(commit_id)
                    with stats_lock:
                        stats["failed"] += 1
                        status = "失败"
                
            except Exception as e:
                commit_hash = commit_mapping[commit_id]['original_commit'].get('hash', '未知')
                print(f"\n[异常] 代码库 {repo_name}, commit {commit_hash}: {str(e)}")
                
                with results_lock:
                    failed_commit_ids.add(commit_id)
                with stats_lock:
                    stats["failed"] += 1
                    status = "异常"
            
            commit_pbar.set_postfix({"状态": status})
            commit_pbar.update(1)
            
            # 更新全局进度
            update_global_progress(global_progress_bar, 1)
    
    commit_pbar.close()
    
    # 输出无diff文件的commit信息
    if no_diff_commits:
        print(f"\n[信息] 代码库 {repo_name}: 发现 {len(no_diff_commits)} 个commit缺少diff.txt文件")
    
    # 4. 严格的结果验证和更新
    updated_commits = []
    
    # 按原始顺序处理每个commit
    for i, original_commit in enumerate(original_commits):
        # 找到对应的commit_id
        matching_commit_id = None
        for cid, mapping in commit_mapping.items():
            if mapping['original_index'] == i:
                matching_commit_id = cid
                break
        
        if not matching_commit_id:
            raise ValueError(f"无法找到索引 {i} 对应的commit_id")
        
        # 克隆原始commit以避免意外修改
        updated_commit = original_commit.copy()
        
        # 如果有处理结果，应用更新
        if matching_commit_id in processing_results:
            result = processing_results[matching_commit_id]
            
            # 最终验证：确保结果属于正确的commit
            if not verify_result_belongs_to_commit(result, updated_commit, matching_commit_id):
                print(f"[警告] 检测到结果与commit不匹配: {matching_commit_id}")
                with stats_lock:
                    stats["failed"] += 1
                    if result.get("skipped", False):
                        stats["skipped"] -= 1
                    else:
                        stats["processed"] -= 1
            else:
                # 安全地应用结果
                updated_commit["optimization_summary_arch"] = result["optimization_summary_arch"]
                updated_commit["optimization_summary_arch_final"] = result["optimization_summary_arch_final"]
        
        updated_commits.append(updated_commit)
    
    # 5. 最终完整性检查
    if len(updated_commits) != len(original_commits):
        raise ValueError(f"数据完整性检查失败：原始{len(original_commits)}个，更新后{len(updated_commits)}个")
    
    # 6. 安全保存文件
    try:
        safe_write_commits(json_file_path, updated_commits)
        
        return {
            "repo_name": repo_name,
            "status": "success",
            "stats": stats,
            "no_diff_commits": no_diff_commits
        }
        
    except Exception as e:
        print(f"\n[错误] 代码库 {repo_name}: 写入文件失败 - {str(e)}")
        return {
            "repo_name": repo_name,
            "status": "write_error",
            "error": str(e),
            "stats": stats,
            "no_diff_commits": no_diff_commits
        }

def main():
    """
    主函数：处理知识库中所有代码库的summary_arch.json文件
    """
    if not os.path.exists(KNOWLEDGE_BASE_ROOT):
        print(f"错误：知识库根目录 {KNOWLEDGE_BASE_ROOT} 不存在")
        return
    
    # 读取代码库列表
    if not os.path.exists(REPO_LIST_FILE):
        print(f"错误：代码库列表文件 {REPO_LIST_FILE} 不存在")
        return
    
    with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
        repositories = json.load(f)
    
    print(f"\n处理知识库: {KNOWLEDGE_BASE_ROOT}")
    print(f"代码库列表: {REPO_LIST_FILE}")
    print(f"每个commit生成 {NUM_GENERATIONS} 个体系结构相关优化策略总结进行投票")
    print(f"使用随机temperature: {TEMPERATURE_MIN} - {TEMPERATURE_MAX}")
    print(f"仓库级别并行数: {MAX_REPO_WORKERS}")
    print(f"commit级别并行数: {MAX_WORKERS}")
    print(f"使用严格验证模式确保数据完整性")
    print(f"从diff.txt文件读取代码差异，跳过缺少diff文件的commit")
    
    if not repositories:
        print("未找到任何代码库")
        return
    
    print(f"找到 {len(repositories)} 个代码库")
    print("正在统计总commit数量...")
    
    # 预统计所有仓库的commit数量
    repo_commit_counts, total_commits = count_total_commits(repositories)
    
    print(f"总计 {total_commits} 个commit需要处理")
    
    # 总体统计
    total_stats = {
        "repositories": len(repositories),
        "success": 0,
        "failed": 0,
        "file_not_found": 0,
        "empty_files": 0,
        "total_commits": total_commits,
        "processed_commits": 0,
        "skipped_commits": 0,
        "failed_commits": 0,
        "no_diff_commits": 0
    }
    
    # 收集所有无diff文件的commit
    all_no_diff_commits = []
    
    # 创建进度条
    # 第一层：总体进度条
    global_progress_bar = tqdm(
        total=total_commits,
        desc="总进度",
        position=0,
        unit="commit"
    )
    
    # 第二层：仓库进度条
    repo_progress_bar = tqdm(
        total=len(repositories),
        desc="仓库进度",
        position=1,
        unit="repo"
    )
    
    # 处理每个代码库 - 使用仓库级别并行
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_REPO_WORKERS) as repo_executor:
        future_to_repo = {}
        position_counter = 2  # 从position=2开始分配给各个仓库的commit进度条
        
        # 提交所有仓库处理任务
        for repo in repositories:
            repo_name = repo.get('name_long', repo.get('name', ''))
            future = repo_executor.submit(
                process_repository_safe,
                repo,
                repo_commit_counts,
                global_progress_bar,
                position_counter
            )
            future_to_repo[future] = repo_name
            position_counter += 1
        
        # 收集结果
        for future in as_completed(future_to_repo):
            repo_name = future_to_repo[future]
            
            try:
                result = future.result()
                results.append(result)
                
                # 线程安全地更新总体统计
                with global_stats_lock:
                    if result["status"] == "success":
                        total_stats["success"] += 1
                        total_stats["processed_commits"] += result["stats"]["processed"]
                        total_stats["skipped_commits"] += result["stats"]["skipped"]
                        total_stats["failed_commits"] += result["stats"]["failed"]
                        total_stats["no_diff_commits"] += result["stats"]["no_diff"]
                        
                        print(f"\n[成功] 代码库 {repo_name}: 总计 {result['stats']['total']} 个commit，处理 {result['stats']['processed']} 个，跳过 {result['stats']['skipped']} 个，失败 {result['stats']['failed']} 个，无diff文件 {result['stats']['no_diff']} 个")
                    elif result["status"] == "file_not_found":
                        total_stats["file_not_found"] += 1
                    elif result["status"] == "empty_file":
                        total_stats["empty_files"] += 1
                    else:
                        total_stats["failed"] += 1
                
                # 线程安全地收集无diff文件的commit信息
                if "no_diff_commits" in result and result["no_diff_commits"]:
                    with global_no_diff_lock:
                        all_no_diff_commits.extend(result["no_diff_commits"])
                
            except Exception as e:
                print(f"\n[异常] 处理代码库 {repo_name} 时发生异常: {str(e)}")
                
                # 创建失败结果
                error_result = {
                    "repo_name": repo_name,
                    "status": "exception",
                    "error": str(e),
                    "stats": {"total": 0, "processed": 0, "skipped": 0, "failed": 0, "no_diff": 0}
                }
                results.append(error_result)
                
                with global_stats_lock:
                    total_stats["failed"] += 1
            
            # 更新仓库进度条
            repo_progress_bar.set_postfix({
                "成功": total_stats["success"],
                "失败": total_stats["failed"] + total_stats["file_not_found"] + total_stats["empty_files"]
            })
            repo_progress_bar.update(1)
    
    # 关闭进度条
    global_progress_bar.close()
    repo_progress_bar.close()
    
    # 打印总体统计
    print(f"\n=== 总体统计 ===")
    print(f"代码库总数: {total_stats['repositories']}")
    print(f"成功处理: {total_stats['success']}")
    print(f"处理失败: {total_stats['failed']}")
    print(f"文件不存在: {total_stats['file_not_found']}")
    print(f"文件为空: {total_stats['empty_files']}")
    print(f"总commit数: {total_stats['total_commits']}")
    print(f"新处理的commit: {total_stats['processed_commits']}")
    print(f"跳过的commit: {total_stats['skipped_commits']}")
    print(f"失败的commit: {total_stats['failed_commits']}")
    print(f"缺少diff文件的commit: {total_stats['no_diff_commits']}")
    
    print(f"\n处理完成！")

if __name__ == "__main__":
    main()

