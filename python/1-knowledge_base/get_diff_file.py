import json
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import difflib
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from tqdm import tqdm

# 全局配置
SKIP_EXISTING = True  # 是否跳过已存在有效diff.txt的commit
MAX_PARALLEL_REPOS = 32  # 控制最大并行处理的仓库数量
MIN_DIFF_SIZE = 10  # 最小diff文件大小（字节），低于此值认为是空文件
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base_all")  # 知识库路径

def is_diff_file_valid(diff_path):
    """
    检查diff.txt文件是否存在且有效
    """
    if not os.path.exists(diff_path):
        return False
    
    try:
        file_size = os.path.getsize(diff_path)
        if file_size < MIN_DIFF_SIZE:
            return False
            
        # 进一步检查文件内容是否有效
        with open(diff_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            # 检查是否包含diff的基本标识或者是否为空
            if not content:
                return False
            # 简单的diff格式检查
            if 'diff --git' not in content and '@@' not in content and len(content.split('\n')) < 2:
                return False
                
        return True
    except Exception:
        return False

def find_before_after_files(commit_dir):
    """
    在commit目录中查找before.*和after.*文件
    返回 (before_file_path, after_file_path, file_extension)
    """
    before_files = glob.glob(os.path.join(commit_dir, "before.*"))
    after_files = glob.glob(os.path.join(commit_dir, "after.*"))
    
    # 过滤掉.txt文件，因为那可能是旧的命名方式
    before_files = [f for f in before_files if not f.endswith('.txt')]
    after_files = [f for f in after_files if not f.endswith('.txt')]
    
    if not before_files or not after_files:
        # 如果没找到，尝试查找.txt文件作为备选
        if not before_files:
            before_txt = os.path.join(commit_dir, "before.txt")
            if os.path.exists(before_txt):
                before_files = [before_txt]
        
        if not after_files:
            after_txt = os.path.join(commit_dir, "after.txt")
            if os.path.exists(after_txt):
                after_files = [after_txt]
    
    if not before_files or not after_files:
        return None, None, None
    
    # 取第一个文件（因为保证只有一个文件被修改）
    before_file = before_files[0]
    after_file = after_files[0]
    
    # 提取文件扩展名，优先从非.txt文件中提取
    if not before_file.endswith('.txt'):
        file_extension = os.path.splitext(before_file)[1]
    elif not after_file.endswith('.txt'):
        file_extension = os.path.splitext(after_file)[1]
    else:
        # 两个都是.txt，默认为.cpp
        file_extension = '.cpp'
    
    return before_file, after_file, file_extension

def generate_diff_from_files(before_file_path, after_file_path, original_filename=None):
    """
    从before和after文件生成unified diff格式的差异
    """
    try:
        # 读取before文件内容
        with open(before_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            before_content = f.readlines()
        
        # 读取after文件内容
        with open(after_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            after_content = f.readlines()
        
        # 如果没有提供原始文件名，尝试从文件推断
        if original_filename is None:
            # 从before文件名推断原始文件名
            before_basename = os.path.basename(before_file_path)
            if before_basename.startswith('before.'):
                original_filename = before_basename[7:]  # 去掉'before.'前缀
            else:
                original_filename = 'modified_file.cpp'  # 默认文件名
        
        # 生成unified diff
        diff_lines = difflib.unified_diff(
            before_content,
            after_content,
            fromfile=f"a/{original_filename}",
            tofile=f"b/{original_filename}",
            lineterm=''
        )
        
        # 将diff转换为字符串
        diff_content = '\n'.join(diff_lines)
        
        # 如果diff为空，说明文件没有差异
        if not diff_content.strip():
            return "# 文件内容相同，无差异\n"
        
        # 添加Git风格的diff header
        git_style_diff = f"diff --git a/{original_filename} b/{original_filename}\n"
        git_style_diff += f"index 0000000..1111111 100644\n"
        git_style_diff += f"--- a/{original_filename}\n"
        git_style_diff += f"+++ b/{original_filename}\n"
        git_style_diff += diff_content
        
        return git_style_diff
        
    except Exception as e:
        print(f"生成diff时发生错误: {str(e)}")
        return None

def find_commits_with_missing_diff(knowledge_base_repo_path):
    """
    查找指定代码库中所有缺失或无效diff.txt的commit
    """
    missing_commits = []
    
    modified_file_path = os.path.join(knowledge_base_repo_path, "modified_file")
    if not os.path.exists(modified_file_path):
        return missing_commits
    
    # 遍历所有commit目录
    for commit_hash in os.listdir(modified_file_path):
        commit_dir = os.path.join(modified_file_path, commit_hash)
        if not os.path.isdir(commit_dir):
            continue
            
        diff_path = os.path.join(commit_dir, "diff.txt")
        
        # 检查是否跳过已存在有效的diff.txt
        if SKIP_EXISTING and is_diff_file_valid(diff_path):
            continue
        
        # 检查diff.txt是否存在且有效
        if not is_diff_file_valid(diff_path):
            # 检查是否存在before和after文件
            before_file, after_file, file_ext = find_before_after_files(commit_dir)
            if before_file and after_file:
                missing_commits.append({
                    'commit_hash': commit_hash,
                    'commit_dir': commit_dir,
                    'diff_path': diff_path,
                    'before_file': before_file,
                    'after_file': after_file,
                    'file_extension': file_ext
                })
    
    return missing_commits

def process_repository(repository_name):
    """
    处理单个代码库的diff.txt修复，用于并行执行
    """
    knowledge_base_repo_path = os.path.join(KNOWLEDGE_BASE_PATH, repository_name)
    
    # 检查知识库目录是否存在
    if not os.path.exists(knowledge_base_repo_path):
        return f"{repository_name}: 知识库目录不存在，跳过。"
    
    # 查找缺失diff的commit
    missing_commits = find_commits_with_missing_diff(knowledge_base_repo_path)
    
    if not missing_commits:
        return f"{repository_name}: 所有commit的diff.txt都完整，无需修复。"
    
    # 初始化统计信息
    total_missing = len(missing_commits)
    fixed_commits = 0
    failed_commits = 0
    skipped_commits = 0
    
    process_log = [f"开始处理代码库: {repository_name}，发现 {total_missing} 个需要修复diff的commit。"]
    
    # 创建当前代码库的进度条
    commit_progress = tqdm(
        missing_commits, 
        desc=f"修复 {repository_name}", 
        unit="commit",
        leave=False,
        position=1
    )
    
    # 逐个修复缺失的diff.txt
    for commit_info in commit_progress:
        commit_hash = commit_info['commit_hash']
        diff_path = commit_info['diff_path']
        before_file = commit_info['before_file']
        after_file = commit_info['after_file']
        file_extension = commit_info['file_extension']
        
        try:
            # 推断原始文件名
            original_filename = f"modified_file{file_extension}" if file_extension else "modified_file.cpp"
            
            # 从before和after文件生成diff内容
            diff_content = generate_diff_from_files(before_file, after_file, original_filename)
            
            if diff_content is not None:
                # 确保目录存在
                os.makedirs(os.path.dirname(diff_path), exist_ok=True)
                
                # 保存diff内容
                with open(diff_path, 'w', encoding='utf-8') as f:
                    f.write(diff_content)
                
                # 验证生成的文件
                if is_diff_file_valid(diff_path):
                    fixed_commits += 1
                else:
                    failed_commits += 1
                    process_log.append(f"commit {commit_hash} 生成的diff.txt无效")
            else:
                failed_commits += 1
                process_log.append(f"无法为commit {commit_hash} 生成diff内容")
                
        except Exception as e:
            failed_commits += 1
            process_log.append(f"处理commit {commit_hash} 时发生异常: {str(e)}")
    
    commit_progress.close()
    
    # 生成处理结果总结
    summary = [
        f"代码库 {repository_name} 处理完成。",
        f"需要修复的commit数量: {total_missing}",
        f"成功修复的commit数量: {fixed_commits}",
        f"修复失败的commit数量: {failed_commits}",
        f"跳过的commit数量: {skipped_commits}",
        "-" * 50
    ]
    
    return "\n".join(process_log + summary)

def fix_missing_diffs_parallel():
    """
    并行修复所有代码库中缺失的diff.txt文件
    """
    # 检查路径是否存在
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"知识库路径不存在: {KNOWLEDGE_BASE_PATH}")
        return
    
    # 获取所有代码库名称
    repository_names = [name for name in os.listdir(KNOWLEDGE_BASE_PATH) 
                       if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, name))]
    
    if not repository_names:
        print("知识库中未找到任何代码库")
        return
    
    print(f"找到 {len(repository_names)} 个代码库待处理，设置并行数为 {MAX_PARALLEL_REPOS}")
    print(f"跳过已存在有效diff.txt: {'是' if SKIP_EXISTING else '否'}")
    print("开始并行修复缺失的diff.txt文件...")
    
    # 统计变量（使用Manager确保并行安全）
    manager = multiprocessing.Manager()
    stats = manager.dict({
        'total_fixed': 0,
        'total_failed': 0,
        'processed_repos': 0
    })
    stats_lock = manager.Lock()
    
    # 使用进程池并行处理多个代码库
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_REPOS) as executor:
        # 提交所有任务并获取future对象
        future_to_repo = {executor.submit(process_repository, repo_name): repo_name 
                          for repo_name in repository_names}
        
        # 使用tqdm显示代码库级别的进度
        repo_progress = tqdm(future_to_repo, desc="处理代码库", unit="repository", position=0)
        
        for future in repo_progress:
            repo_name = future_to_repo[future]
            try:
                result = future.result()
                print(result)  # 打印每个代码库的处理结果
                
                # 线程安全地更新统计信息
                with stats_lock:
                    stats['processed_repos'] += 1
                    # 从结果中提取统计数据
                    if "成功修复的commit数量:" in result:
                        lines = result.split('\n')
                        for line in lines:
                            if "成功修复的commit数量:" in line:
                                fixed = int(line.split(':')[1].strip())
                                stats['total_fixed'] += fixed
                            elif "修复失败的commit数量:" in line:
                                failed = int(line.split(':')[1].strip())
                                stats['total_failed'] += failed
                
            except Exception as e:
                print(f"代码库 {repo_name} 处理时发生异常: {str(e)}")
                with stats_lock:
                    stats['total_failed'] += 1
    
    repo_progress.close()
    
    # 输出总体统计
    print("\n" + "="*60)
    print("总体修复统计:")
    print(f"处理的代码库数量: {stats['processed_repos']}/{len(repository_names)}")
    print(f"总共成功修复的commit数量: {stats['total_fixed']}")
    print(f"总共修复失败的commit数量: {stats['total_failed']}")
    total_attempts = stats['total_fixed'] + stats['total_failed']
    if total_attempts > 0:
        print(f"总体修复成功率: {(stats['total_fixed']/total_attempts*100):.1f}%")
    print("="*60)

def preview_missing_diffs():
    """
    预览模式：统计缺失diff.txt的情况，不进行修复
    """
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"知识库路径不存在: {KNOWLEDGE_BASE_PATH}")
        return
    
    repository_names = [name for name in os.listdir(KNOWLEDGE_BASE_PATH) 
                       if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, name))]
    
    if not repository_names:
        print("知识库中未找到任何代码库")
        return
    
    total_missing = 0
    total_commits = 0
    repo_stats = []
    
    print("扫描知识库中缺失diff.txt的commit...")
    
    # 使用tqdm显示扫描进度
    for repo_name in tqdm(repository_names, desc="扫描代码库", unit="repository"):
        repo_knowledge_path = os.path.join(KNOWLEDGE_BASE_PATH, repo_name)
        missing_commits = find_commits_with_missing_diff(repo_knowledge_path)
        
        # 统计该代码库总commit数
        modified_file_path = os.path.join(repo_knowledge_path, "modified_file")
        repo_total_commits = 0
        if os.path.exists(modified_file_path):
            repo_total_commits = len([name for name in os.listdir(modified_file_path) 
                                    if os.path.isdir(os.path.join(modified_file_path, name))])
        
        total_commits += repo_total_commits
        
        if missing_commits:
            repo_stats.append({
                'repo_name': repo_name,
                'missing_count': len(missing_commits),
                'total_commits': repo_total_commits
            })
            total_missing += len(missing_commits)
        elif repo_total_commits > 0:
            repo_stats.append({
                'repo_name': repo_name,
                'missing_count': 0,
                'total_commits': repo_total_commits
            })
    
    # 输出统计结果
    print("\n" + "="*60)
    print("缺失diff.txt统计报告:")
    print(f"扫描的代码库数量: {len(repository_names)}")
    print(f"总commit数量: {total_commits}")
    print(f"缺失diff.txt的commit数量: {total_missing}")
    if total_commits > 0:
        print(f"缺失比例: {(total_missing/total_commits*100):.1f}%")
    
    if repo_stats:
        print(f"\n存在commit的代码库数量: {len([r for r in repo_stats if r['total_commits'] > 0])}")
        print(f"存在缺失diff.txt的代码库数量: {len([r for r in repo_stats if r['missing_count'] > 0])}")
        
        print("\n各代码库详细统计:")
        # 按缺失数量排序
        repo_stats.sort(key=lambda x: x['missing_count'], reverse=True)
        for stat in repo_stats:
            if stat['total_commits'] > 0:
                missing_ratio = (stat['missing_count']/stat['total_commits']*100) if stat['total_commits'] > 0 else 0
                print(f"  {stat['repo_name']}: {stat['missing_count']}/{stat['total_commits']} ({missing_ratio:.1f}%)")
    
    print("="*60)

# 程序入口
if __name__ == "__main__":
    # 可以通过修改这里来选择运行模式
    PREVIEW_MODE = False  # 设置为True则只预览，False则执行修复
    
    if PREVIEW_MODE:
        preview_missing_diffs()
    else:
        fix_missing_diffs_parallel()