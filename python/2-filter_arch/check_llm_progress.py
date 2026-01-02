import json
import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 配置
KNOWLEDGE_BASE_ROOT = os.path.join(config.root_path, "knowledge_base")
LLM_PROCESSING_FILE = "is_opt_arch_llm.json"
LLM_RESULT_FIELD = "is_opt_arch_llm"
repo_list_file = os.path.join(config.root_path, "repo_list_30342.json")
MAX_WORKERS = 128  # 并行读取仓库的线程数


def check_single_repo_progress(repo):
    """
    检查单个仓库的LLM处理进度
    返回统计信息
    """
    repo_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo)
    llm_file = os.path.join(repo_path, LLM_PROCESSING_FILE)
    
    # 检查文件是否存在
    if not os.path.exists(llm_file):
        return {
            "repo": repo,
            "status": "no_file",
            "total": 0,
            "processed": 0,
            "pending": 0,
            "passed": 0,
            "failed": 0,
            "unknown": 0
        }
    
    try:
        # 读取文件
        with open(llm_file, "r", encoding="utf-8") as f:
            commits = json.load(f)
        
        total = len(commits)
        passed = 0
        failed = 0
        unknown = 0
        
        # 统计各种状态
        for commit in commits:
            result = commit.get(LLM_RESULT_FIELD, "unknown")
            if result == "true":
                passed += 1
            elif result == "false":
                failed += 1
            else:
                unknown += 1
        
        processed = passed + failed
        pending = unknown
        
        return {
            "repo": repo,
            "status": "success",
            "total": total,
            "processed": processed,
            "pending": pending,
            "passed": passed,
            "failed": failed,
            "unknown": unknown
        }
    except Exception as e:
        return {
            "repo": repo,
            "status": "error",
            "error": str(e),
            "total": 0,
            "processed": 0,
            "pending": 0,
            "passed": 0,
            "failed": 0,
            "unknown": 0
        }


def check_llm_progress(repositories):
    """
    检查所有仓库的LLM处理进度
    """
    print("=" * 60)
    print("📊 LLM处理进度统计")
    print("=" * 60)
    print(f"📦 待检查代码库数量: {len(repositories)}")
    print(f"⚡ 并行数: {MAX_WORKERS}")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for repo in repositories:
            future = executor.submit(check_single_repo_progress, repo)
            futures.append(future)
        
        # 收集结果
        all_results = []
        for future in tqdm(futures, desc="🔍 检查进度", unit="repo"):
            result = future.result()
            all_results.append(result)
    
    # 汇总统计
    repos_with_file = 0
    repos_no_file = 0
    repos_error = 0
    
    total_commits = 0
    processed_commits = 0
    pending_commits = 0
    passed_commits = 0
    failed_commits = 0
    unknown_commits = 0
    
    repos_with_pending = []
    repos_fully_processed = []
    
    for result in all_results:
        if result["status"] == "success":
            repos_with_file += 1
            total_commits += result["total"]
            processed_commits += result["processed"]
            pending_commits += result["pending"]
            passed_commits += result["passed"]
            failed_commits += result["failed"]
            unknown_commits += result["unknown"]
            
            if result["pending"] > 0:
                repos_with_pending.append({
                    "repo": result["repo"],
                    "total": result["total"],
                    "pending": result["pending"],
                    "processed": result["processed"]
                })
            elif result["total"] > 0:
                repos_fully_processed.append(result["repo"])
        elif result["status"] == "no_file":
            repos_no_file += 1
        else:
            repos_error += 1
    
    # 打印详细统计
    print("\n" + "=" * 60)
    print("📊 代码库统计:")
    print(f"  ✅ 有LLM处理文件: {repos_with_file} 个代码库")
    print(f"  ❌ 无LLM处理文件: {repos_no_file} 个代码库")
    print(f"  ⚠️  读取错误: {repos_error} 个代码库")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("📊 Commit处理统计:")
    print(f"  📝 总Commit数: {total_commits} 个")
    print(f"  ✅ 已处理: {processed_commits} 个 (包括通过和未通过)")
    print(f"  ⏳ 待处理: {pending_commits} 个")
    if total_commits > 0:
        progress = (processed_commits / total_commits) * 100
        print(f"  📈 处理进度: {progress:.2f}%")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("📊 LLM判断结果统计:")
    print(f"  ✅ 通过 (true): {passed_commits} 个")
    print(f"  ❌ 未通过 (false): {failed_commits} 个")
    print(f"  ❓ 未知 (unknown): {unknown_commits} 个")
    if processed_commits > 0:
        pass_rate = (passed_commits / processed_commits) * 100
        print(f"  📈 通过率: {pass_rate:.2f}%")
    print("=" * 60)
    
    # 打印有待处理commits的仓库（Top 20）
    if repos_with_pending:
        print("\n" + "=" * 60)
        print(f"📋 有待处理commits的仓库 (共{len(repos_with_pending)}个):")
        print("   Top 20 按待处理数量排序:")
        print("=" * 60)
        repos_with_pending.sort(key=lambda x: x["pending"], reverse=True)
        for i, repo_info in enumerate(repos_with_pending[:20], 1):
            repo = repo_info["repo"]
            total = repo_info["total"]
            pending = repo_info["pending"]
            processed = repo_info["processed"]
            progress = (processed / total) * 100 if total > 0 else 0
            print(f"  {i}. {repo}")
            print(f"     总数: {total}, 已处理: {processed}, 待处理: {pending}, 进度: {progress:.1f}%")
    
    # 打印完全处理完的仓库数量
    if repos_fully_processed:
        print("\n" + "=" * 60)
        print(f"✅ 完全处理完的仓库: {len(repos_fully_processed)} 个")
        print("=" * 60)
    
    # 估算剩余时间（假设处理速度）
    if pending_commits > 0 and processed_commits > 0:
        print("\n" + "=" * 60)
        print("⏱️  估算信息:")
        print(f"  如果以当前速度继续处理，还需要处理 {pending_commits} 个commits")
        print("  (实际处理时间取决于LLM API速度和并行配置)")
        print("=" * 60)


if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_BASE_ROOT):
        print(f"❌ 错误: 目录 '{KNOWLEDGE_BASE_ROOT}' 不存在。")
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
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到代码库列表文件 {repo_list_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误：解析 JSON 文件失败 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误：读取代码库列表时发生错误 - {e}")
        sys.exit(1)
    
    if not repositories:
        print("❌ 未找到任何代码仓库")
        sys.exit(1)
    
    # 执行进度检查
    check_llm_progress(repositories)
    
    print("\n" + "=" * 60)
    print("✅ 进度检查完成！")
    print("=" * 60)

