import json
import os
import time
import shutil
from git import Repo
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# ================= 配置区 =================
# 全局变量，控制是否跳过已经处理过的代码库
SKIP_EXISTING_ARCH_RESULTS = False
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base")

# 设置代码库列表文件路径
REPO_LIST_FILE = os.path.join(config.root_path, "repo_list_30342.json")

# 并行处理配置
MAX_WORKERS = 256  # 最大并行处理的代码库数量
# ========================================


# 架构相关优化关键词
architecture_keywords = [
    # CPU 架构名称
    "x86", "x86_64", "x86-64", "amd64",
    "arm", "arm64", "aarch64", "armv7", "armv8",
    "riscv", "risc-v", "riscv64",
    "powerpc", "ppc", "ppc64",
    "mips", "mips64",
    "sparc",
    "loongarch",
    "s390x",
    
    # SIMD 和向量化
    "simd",
    "sse", "sse2", "sse3", "ssse3", "sse4", "sse4.1", "sse4.2",
    "avx", "avx2", "avx-512", "avx512",
    "neon",  # ARM SIMD
    "altivec",  # PowerPC SIMD
    "msa",  # MIPS SIMD
    "vector", "vectorize", "vectorization",
    
    # 缓存优化
    "cache line",
    "cache align", "cache-align", "cacheline",
    "cache friendly",
    "cache coherence", "cache coherency",
    "prefetch", "prefetching",
    "false sharing",
    "l1 cache", "l2 cache", "l3 cache",
    "cache miss", "cache hit",
    "cache warmup",
    
    # 内存架构
    "memory align", "memory alignment", "aligned memory",
    "numa",
    "memory order", "memory ordering",
    "memory barrier", "memory fence",
    "memory model",
    "huge page", "hugepage",
    "tlb",
    
    # 原子操作和同步
    "atomic", "atomics",
    "compare-and-swap", "cas", "cmpxchg",
    "load-acquire", "store-release",
    "relaxed ordering",
    "memory_order",
    
    # 编译器和指令相关
    "intrinsic", "intrinsics",
    "inline asm", "inline assembly", "__asm__",
    "builtin",
    "instruction", "instructions",
    "cpu feature", "cpu features",
    "arch-specific", "architecture-specific",
    
    # 特定指令和技术
    "branch prediction",
    "pipeline", "pipelining",
    "instruction level parallelism", "ilp",
    "out-of-order",
    "speculative execution",
    "hardware prefetch",
    
    # 架构特定优化
    "arch optimization",
    "platform specific",
    "cpu specific",
    "architecture dependent",
    "cross-platform",
    "portable",
    
    # 性能计数器和性能监控
    "perf counter",
    "pmu",  # Performance Monitoring Unit
    "hardware counter",
    
    # 其他架构相关
    "endianness", "little-endian", "big-endian",
    "word size",
    "register", "registers",
    "calling convention",
    "abi",  # Application Binary Interface
]


def process_single_repo_keyword(repo, keywords):
    """处理单个代码库的关键词筛选"""
    knowledge_base_path = os.path.join(KNOWLEDGE_BASE_PATH, repo)
    input_file = os.path.join(knowledge_base_path, "is_opt_final.json")
    output_arch = os.path.join(knowledge_base_path, "is_opt_arch_keyword.json")
    
    # 检查是否需要跳过当前代码库
    if SKIP_EXISTING_ARCH_RESULTS and os.path.exists(output_arch):
        return {"repo": repo, "status": "skipped", "reason": "目标文件已存在"}
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        return {"repo": repo, "status": "error", "reason": "输入文件不存在"}
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_arch), exist_ok=True)
    
    try:
        result = filter_commits_by_arch_keywords(input_file, output_arch, keywords, repo)
        return {
            "repo": repo,
            "status": "success",
            "total_commits": result["total_commits"],
            "passed_commits": result["filtered_commits"],
            "failed_commits": result["total_commits"] - result["filtered_commits"]
        }
    except Exception as e:
        return {"repo": repo, "status": "error", "reason": str(e)}


def process_arch_keywords_phase(repositories, keywords):
    """
    针对所有代码库，从is_opt_final.json读取commit并执行架构关键词筛选，
    如果对应的 is_opt_arch_keyword.json 已存在，则根据配置决定是否跳过。
    支持并行处理多个代码库。
    """
    print("=" * 60)
    print("🚀 开始架构关键词筛选阶段")
    print("=" * 60)
    print(f"📊 待处理代码库数量: {len(repositories)}")
    print(f"⚡ 最大并行数: {MAX_WORKERS}")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for repo in repositories:
            future = executor.submit(process_single_repo_keyword, repo, keywords)
            futures.append((future, repo))
        
        # 收集结果并显示进度
        success_count = 0
        skipped_count = 0
        error_count = 0
        input_not_exist_count = 0
        other_error_count = 0
        
        total_commits = 0
        passed_commits = 0
        failed_commits = 0
        
        for future, repo in tqdm(futures, desc="📦 筛选架构相关commit", unit="repo"):
            try:
                result = future.result()
                if result["status"] == "success":
                    success_count += 1
                    total_commits += result.get("total_commits", 0)
                    passed_commits += result.get("passed_commits", 0)
                    failed_commits += result.get("failed_commits", 0)
                elif result["status"] == "skipped":
                    skipped_count += 1
                elif result["status"] == "error":
                    error_count += 1
                    if result.get("reason") == "输入文件不存在":
                        input_not_exist_count += 1
                    else:
                        other_error_count += 1
            except Exception as e:
                print(f"❌ 处理仓库 {repo} 时发生异常: {e}")
                error_count += 1
                other_error_count += 1
    
    # 打印详细统计信息
    print("\n" + "=" * 60)
    print("📊 代码库处理统计:")
    print(f"  ✅ 成功处理: {success_count} 个代码库")
    print(f"  ⏭️  跳过: {skipped_count} 个代码库")
    print(f"  ❌ 失败: {error_count} 个代码库")
    if error_count > 0:
        print(f"     └─ 输入文件不存在: {input_not_exist_count} 个")
        print(f"     └─ 其他错误: {other_error_count} 个")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("📊 Commit处理统计:")
    print(f"  📝 总共处理: {total_commits} 个commit")
    print(f"  ✅ 通过关键词筛选: {passed_commits} 个commit")
    print(f"  ❌ 未通过筛选: {failed_commits} 个commit")
    if total_commits > 0:
        pass_rate = (passed_commits / total_commits) * 100
        print(f"  📈 通过率: {pass_rate:.2f}%")
    print("=" * 60)


def filter_commits_by_arch_keywords(input_file, output_file, keywords, repo_name):
    """
    从is_opt_final.json文件读取commit信息，
    使用架构相关关键词匹配来筛选出与架构相关的优化commit，
    保存结果到is_opt_arch_keyword.json。
    返回统计信息。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件 '{input_file}' 不存在")
    
    # 读取is_opt_final.json中的commit信息
    with open(input_file, "r", encoding="utf-8") as file:
        all_commits = json.load(file)
    
    # 筛选包含架构关键词的commit
    arch_commits = []
    for commit in all_commits:
        message = commit.get("message", "").lower()
        if any(keyword.lower() in message for keyword in keywords):
            # 保留原始commit的所有信息，并添加一个标记
            commit_copy = commit.copy()
            commit_copy["contains_optimization_keyword"] = True
            arch_commits.append(commit_copy)
    
    # 将筛选结果写入输出文件
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(arch_commits, file, indent=4, ensure_ascii=False)
    
    return {
        "total_commits": len(all_commits),
        "filtered_commits": len(arch_commits)
    }


if __name__ == "__main__":
    # 从 JSON 文件中读取代码库列表
    try:
        with open(REPO_LIST_FILE, 'r', encoding='utf-8') as f:
            repo_list_data = json.load(f)
        
        # 根据 repo_list_c.json 的结构提取代码库名称
        # 文件结构：包含 name、http_url、ssh_url、stars 字段的对象数组
        if isinstance(repo_list_data, list):
            # 提取每个对象的 name 字段
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
        
        print(f"📊 从 {REPO_LIST_FILE} 读取到 {len(repositories)} 个代码库")
        print(f"📦 代码库列表: {', '.join(repositories[:5])}{'...' if len(repositories) > 5 else ''}")
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到代码库列表文件 {REPO_LIST_FILE}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误：解析 JSON 文件失败 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误：读取代码库列表时发生错误 - {e}")
        sys.exit(1)

    # 执行架构关键词筛选
    process_arch_keywords_phase(repositories, architecture_keywords)

    print("\n" + "=" * 60)
    print("✅ 所有仓库处理完成！")
    print("=" * 60)