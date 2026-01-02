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

# 全局变量，控制是否跳过已经处理过的代码库
SKIP_EXISTING_KEYWORD_RESULTS = False
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base_all")

# 并行处理配置
MAX_WORKERS = 8  # 可以根据CPU核心数调整

# 体系结构相关关键词
architecture_keywords = ["x86"]
OUTPUT_FILE_NAME = "is_opt_keyword_x86.json"
# architecture_keywords = ["arm64"]
# OUTPUT_FILE_NAME = "is_opt_keyword_arm64.json"
# architecture_keywords = ["x86", "arm64"]
# OUTPUT_FILE_NAME = "is_opt_keyword_arch_all.json"

# 优化相关关键词
optimization_keywords = [
    # 核心性能指标关键词
    "performance",
    "optimize", "optimization",
    "speedup", "speed up", "speed-up",
    "fast", "faster", "fastest",
    "efficient", "efficiency",
    "throughput",
    "latency", "low-latency",
    
    # 优化动作关键词
    "accelerate", "acceleration",
    "improve", "improvement",
    "enhance", "enhancement",
    "boost", "boosted",
    "tune", "tuning",
    "refactor for speed",
    "perf gain",
    "perf win",
    
    # 资源使用优化关键词
    "reduce overhead",
    "reduce memory",
    "memory usage",
    "memory footprint",
    "memory consumption",
    "reduce allocation",
    "cache friendly",
    "cache efficiency",
    "cache hit",
    "cache miss",
    
    # 具体优化技术
    "inlining", "inline function",
    "loop unrolling",
    "vectorization", "vectorize",
    "simd",
    "parallelization", "parallelize",
    "multithreading", "multi-threading",
    "memoization",
    "lazy evaluation",
    "lock-free",
    "zero-copy",
    "hot path",
    "branch prediction",
    "prefetch",
    
    # 性能问题修复
    "bottleneck",
    "hotspot",
    "slow path",
    "critical path",
    "profile guided",
    "profiler",
    "complexity",
    
    # 特定领域优化
    "buffer reuse",
    "avoid copy",
    "avoid allocation",
    "reduce copy",
    "batch processing"
]

def process_single_folder(folder, opt_keywords, arch_keywords):
    """
    处理单个文件夹的函数，用于并行执行
    返回：(folder_name, success, message, filtered_count, total_count)
    """
    try:
        knowledge_base_path = os.path.join(KNOWLEDGE_BASE_PATH, folder)
        c_language_file = os.path.join(knowledge_base_path, "c_language.json")
        output_keyword = os.path.join(knowledge_base_path, OUTPUT_FILE_NAME)
        
        # 检查c_language.json是否存在
        if not os.path.exists(c_language_file):
            return (folder, False, f"文件 '{c_language_file}' 不存在", 0, 0)
            
        # 检查是否需要跳过当前代码库
        if SKIP_EXISTING_KEYWORD_RESULTS and os.path.exists(output_keyword):
            return (folder, True, f"文件 '{output_keyword}' 已存在，跳过", 0, 0)
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_keyword), exist_ok=True)
        
        # 执行关键词筛选
        filtered_count, total_count = filter_commits_by_keywords(c_language_file, output_keyword, opt_keywords, arch_keywords, folder)
        
        return (folder, True, f"处理成功", filtered_count, total_count)
        
    except Exception as e:
        return (folder, False, f"处理时发生错误: {str(e)}", 0, 0)

def process_keywords_phase_parallel(knowledge_base_folders, opt_keywords, arch_keywords):
    """
    使用并行处理的关键词筛选阶段
    """
    print("===== 第一阶段：关键词筛选（并行处理） =====")
    print(f"优化关键词数量: {len(opt_keywords)}")
    print(f"体系结构关键词数量: {len(arch_keywords)}")
    print(f"体系结构关键词: {', '.join(arch_keywords)}")
    print(f"并行工作线程数: {MAX_WORKERS}")
    
    # 预筛选需要处理的文件夹
    folders_to_process = []
    for folder in knowledge_base_folders:
        c_language_file = os.path.join(KNOWLEDGE_BASE_PATH, folder, "c_language.json")
        output_keyword = os.path.join(KNOWLEDGE_BASE_PATH, folder, OUTPUT_FILE_NAME)
        
        if not os.path.exists(c_language_file):
            print(f"[Info] 跳过文件夹 {folder}：未找到 c_language.json 文件")
            continue
            
        if SKIP_EXISTING_KEYWORD_RESULTS and os.path.exists(output_keyword):
            print(f"[Info] 跳过文件夹 {folder}：输出文件已存在")
            continue
            
        folders_to_process.append(folder)
    
    print(f"需要处理的文件夹数量: {len(folders_to_process)}")
    
    if not folders_to_process:
        print("没有需要处理的文件夹")
        return
    
    # 统计变量
    total_processed = 0
    total_filtered = 0
    total_commits = 0
    successful_folders = 0
    failed_folders = 0
    
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_folder = {
            executor.submit(process_single_folder, folder, opt_keywords, arch_keywords): folder
            for folder in folders_to_process
        }
        
        # 使用tqdm显示进度条
        with tqdm(total=len(folders_to_process), desc="Processing folders") as pbar:
            for future in as_completed(future_to_folder):
                folder_name, success, message, filtered_count, total_count = future.result()
                
                if success:
                    successful_folders += 1
                    total_filtered += filtered_count
                    total_commits += total_count
                    if filtered_count > 0:  # 只有找到筛选结果时才显示详细信息
                        tqdm.write(f"[Success] {folder_name}: 从 {total_count} 个commit中找到 {filtered_count} 个匹配的commit")
                else:
                    failed_folders += 1
                    tqdm.write(f"[Error] {folder_name}: {message}")
                
                total_processed += 1
                pbar.update(1)
                
                # 更新进度条描述
                pbar.set_postfix({
                    'Success': successful_folders,
                    'Failed': failed_folders,
                    'Filtered': total_filtered
                })
    
    # 输出最终统计信息
    print(f"\n===== 处理完成统计 =====")
    print(f"总共处理文件夹数: {total_processed}")
    print(f"成功处理: {successful_folders}")
    print(f"处理失败: {failed_folders}")
    print(f"总共扫描的commit数: {total_commits}")
    print(f"筛选出的commit数: {total_filtered}")
    if total_commits > 0:
        print(f"筛选率: {total_filtered/total_commits*100:.2f}%")

def process_keywords_phase(knowledge_base_folders, opt_keywords, arch_keywords):
    """
    串行处理版本（保留作为备选）
    """
    print("===== 第一阶段：关键词筛选（串行处理） =====")
    print(f"优化关键词数量: {len(opt_keywords)}")
    print(f"体系结构关键词数量: {len(arch_keywords)}")
    print(f"体系结构关键词: {', '.join(arch_keywords)}")
    
    for folder in tqdm(knowledge_base_folders, desc="Keyword filtering"):
        knowledge_base_path = os.path.join(KNOWLEDGE_BASE_PATH, folder)
        c_language_file = os.path.join(knowledge_base_path, "c_language.json")
        output_keyword = os.path.join(knowledge_base_path, OUTPUT_FILE_NAME)
        
        # 检查c_language.json是否存在
        if not os.path.exists(c_language_file):
            print(f"[Keyword] 警告：文件 '{c_language_file}' 不存在，跳过文件夹 {folder}。")
            continue
            
        # 检查是否需要跳过当前代码库
        if SKIP_EXISTING_KEYWORD_RESULTS and os.path.exists(output_keyword):
            print(f"[Keyword] 文件 '{output_keyword}' 已存在，跳过代码库 {folder}。")
            continue
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_keyword), exist_ok=True)
        filter_commits_by_keywords(c_language_file, output_keyword, opt_keywords, arch_keywords, folder)

def filter_commits_by_keywords(input_file, output_file, opt_keywords, arch_keywords, repo_name):
    """
    从c_language.json文件读取commit信息，
    使用关键词匹配来筛选出实现性能优化且涉及特定体系结构的commit，
    保存结果到 OUTPUT_FILE_NAME
    
    筛选条件：
    1. commit message 中必须包含至少一个优化关键词
    2. commit message 中必须包含至少一个体系结构关键词
    
    返回：(filtered_count, total_count)
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"[Keyword] 错误：输入文件 '{input_file}' 不存在。")
            return 0, 0
            
        # 读取c_language.json中的commit信息
        with open(input_file, "r", encoding="utf-8") as file:
            all_commits = json.load(file)
            
        # 筛选同时包含优化关键词和体系结构关键词的commit
        filtered_commits = []
        for commit in all_commits:
            message = commit.get("message", "").lower()
            
            # 检查是否包含优化关键词
            has_optimization_keyword = any(keyword.lower() in message for keyword in opt_keywords)
            
            # 检查是否包含体系结构关键词
            has_architecture_keyword = any(keyword.lower() in message for keyword in arch_keywords)
            
            # 只有同时满足两个条件的commit才会被筛选出来
            if has_optimization_keyword and has_architecture_keyword:
                # 保留原始commit的所有信息，并添加标记
                commit_copy = commit.copy()
                commit_copy["contains_optimization_keyword"] = True
                commit_copy["contains_architecture_keyword"] = True
                
                # 记录匹配到的具体关键词（可选，用于调试）
                matched_opt_keywords = [kw for kw in opt_keywords if kw.lower() in message]
                matched_arch_keywords = [kw for kw in arch_keywords if kw.lower() in message]
                commit_copy["matched_optimization_keywords"] = matched_opt_keywords
                commit_copy["matched_architecture_keywords"] = matched_arch_keywords
                
                filtered_commits.append(commit_copy)
                
        # 将筛选结果写入输出文件
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(filtered_commits, file, indent=4)
            
        # 返回统计信息而不是直接打印（在并行模式下避免输出混乱）
        return len(filtered_commits), len(all_commits)
        
    except Exception as e:
        print(f"[Keyword] 处理 {repo_name} 时发生错误: {str(e)}")
        return 0, 0

def check_keyword_distribution(knowledge_base_folders):
    """
    统计各个关键词的分布情况，用于分析和调试
    """
    print("\n===== 关键词分布统计 =====")
    total_commits = 0
    opt_only_commits = 0
    arch_only_commits = 0
    both_commits = 0
    
    for folder in tqdm(knowledge_base_folders, desc="Analyzing keyword distribution"):
        c_language_file = os.path.join(KNOWLEDGE_BASE_PATH, folder, "c_language.json")
        if not os.path.exists(c_language_file):
            continue
            
        try:
            with open(c_language_file, "r", encoding="utf-8") as file:
                all_commits = json.load(file)
                
            for commit in all_commits:
                total_commits += 1
                message = commit.get("message", "").lower()
                
                has_opt = any(keyword.lower() in message for keyword in optimization_keywords)
                has_arch = any(keyword.lower() in message for keyword in architecture_keywords)
                
                if has_opt and has_arch:
                    both_commits += 1
                elif has_opt:
                    opt_only_commits += 1
                elif has_arch:
                    arch_only_commits += 1
                    
        except Exception as e:
            print(f"[Stats] 处理 {folder} 时发生错误: {str(e)}")
    
    print(f"总计commit数量: {total_commits}")
    if total_commits > 0:
        print(f"仅包含优化关键词: {opt_only_commits} ({opt_only_commits/total_commits*100:.2f}%)")
        print(f"仅包含体系结构关键词: {arch_only_commits} ({arch_only_commits/total_commits*100:.2f}%)")
        print(f"同时包含两类关键词: {both_commits} ({both_commits/total_commits*100:.2f}%)")

if __name__ == "__main__":
    # 检查knowledge_base目录是否存在
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"Error: 目录 '{KNOWLEDGE_BASE_PATH}' 不存在。")
        sys.exit(1)
    
    # 获取knowledge_base文件夹下的所有子文件夹
    knowledge_base_folders = [
        folder for folder in os.listdir(KNOWLEDGE_BASE_PATH)
        if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, folder))
    ]
    
    # 进一步筛选：只处理包含c_language.json文件的文件夹
    valid_folders = []
    for folder in knowledge_base_folders:
        c_language_file = os.path.join(KNOWLEDGE_BASE_PATH, folder, "c_language.json")
        if os.path.exists(c_language_file):
            valid_folders.append(folder)
        else:
            print(f"[Info] 跳过文件夹 {folder}：未找到 c_language.json 文件")
    
    print(f"找到 {len(valid_folders)} 个包含 c_language.json 的文件夹")
    
    if not valid_folders:
        print("没有找到任何包含 c_language.json 的文件夹，程序退出。")
        sys.exit(1)
    
    # 可选：显示关键词分布统计
    # check_keyword_distribution(valid_folders)
    
    # 第一阶段：关键词筛选（使用并行处理）
    process_keywords_phase_parallel(valid_folders, optimization_keywords, architecture_keywords)
    
    # 如果需要使用串行处理，可以注释上面一行，取消注释下面一行
    # process_keywords_phase(valid_folders, optimization_keywords, architecture_keywords)
    
    print("\n所有仓库处理完成！")