import json
import os
import subprocess
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from tqdm import tqdm

# 全局配置
SKIP_EXISTING = True  # 控制是否跳过已处理的commit
MAX_PARALLEL_REPOS = 64  # 控制最大并行处理的仓库数量

def decode_file_content(content, file_path, commit_hash):
    """
    尝试用 UTF-8、ISO-8859-1 和 Windows-1252 解码文件内容。
    如果解码成功，返回解码后的字符串；否则返回 None。
    """
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
    for encoding in encodings:
        try:
            decoded_content = content.decode(encoding)
            return decoded_content
        except UnicodeDecodeError:
            continue
    print(f"文件 {file_path} 在 commit {commit_hash} 中无法用 UTF-8、ISO-8859-1 或 Windows-1252 解码，跳过。")
    return None

def process_repository(repository_name):
    """处理单个代码库的函数，用于并行执行"""
    # 根路径
    root_path = config.root_path
    repository_base_path = os.path.join(root_path, "repository")
    result_base_path = os.path.join(root_path, "knowledge_base")
    
    repository_path = os.path.join(repository_base_path, repository_name)
    result_path = os.path.join(result_base_path, repository_name)
    json_file_path = os.path.join(result_path, "is_opt_keyword.json")
    output_base_path = os.path.join(result_path, "modified_file")
    
    # 检查是否是代码库目录
    if not os.path.isdir(repository_path):
        return f"{repository_name}: 不是有效的代码库目录，跳过。"
    
    # 检查是否存在 is_opt_keyword.json
    if not os.path.exists(json_file_path):
        return f"{repository_name}: is_opt_keyword.json 不存在，跳过。"
    
    # 读取 json 文件
    try:
        with open(json_file_path, 'r') as f:
            commits = json.load(f)
    except Exception as e:
        return f"{repository_name}: 读取 JSON 文件失败: {str(e)}"
    
    # 初始化统计信息
    total_commits = len(commits)
    processed_commits = 0
    skipped_commits = 0
    failed_commits = 0
    
    process_log = [f"开始处理代码库: {repository_name}，共 {total_commits} 个 commit。"]
    
    # 遍历每个 commit
    for commit in commits:
        commit_hash = commit['hash']
        modified_files = commit['modified_files']
        output_dir = os.path.join(output_base_path, commit_hash)
        
        # 检查是否跳过已处理的commit
        if SKIP_EXISTING and os.path.exists(output_dir):
            before_path = os.path.join(output_dir, 'before.txt')
            after_path = os.path.join(output_dir, 'after.txt')
            diff_path = os.path.join(output_dir, 'diff.txt')
            
            if os.path.exists(before_path) and os.path.exists(after_path) and os.path.exists(diff_path):
                skipped_commits += 1
                continue
        
        try:
            # 获取 diff 信息（以二进制模式读取）
            diff_output_path = os.path.join(output_dir, 'diff.txt')
            git_show_command = f"git -C {repository_path} show {commit_hash} --pretty=format:'' --patch"
            diff_output = subprocess.check_output(git_show_command, shell=True, stderr=subprocess.PIPE)
            
            # 尝试解码 diff 输出
            diff_output_decoded = decode_file_content(diff_output, "diff", commit_hash)
            if diff_output_decoded is None:
                process_log.append(f"commit {commit_hash} 的 diff 输出无法解码，跳过。")
                failed_commits += 1
                continue
            
            # 遍历每个被修改的文件
            for file_path in modified_files:
                try:
                    # 检查文件在父 commit 中是否存在
                    check_existence_command = f"git -C {repository_path} ls-tree {commit_hash}^ {file_path}"
                    subprocess.check_output(check_existence_command, shell=True, stderr=subprocess.PIPE)
                    
                    # 确保文件夹存在
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 获取修改前的文件内容（以二进制模式读取）
                    before_content = subprocess.check_output(
                        f"git -C {repository_path} show {commit_hash}^:{file_path}",
                        shell=True, stderr=subprocess.PIPE
                    )
                    before_path = os.path.join(output_dir, 'before.txt')
                    before_content_decoded = decode_file_content(before_content, file_path, commit_hash)
                    if before_content_decoded is not None:
                        with open(before_path, 'w') as before_file:
                            before_file.write(before_content_decoded)
                    
                    # 获取修改后的文件内容（以二进制模式读取）
                    after_content = subprocess.check_output(
                        f"git -C {repository_path} show {commit_hash}:{file_path}",
                        shell=True, stderr=subprocess.PIPE
                    )
                    after_path = os.path.join(output_dir, 'after.txt')
                    after_content_decoded = decode_file_content(after_content, file_path, commit_hash)
                    if after_content_decoded is not None:
                        with open(after_path, 'w') as after_file:
                            after_file.write(after_content_decoded)
                except subprocess.CalledProcessError as e:
                    error_message = e.stderr.decode('utf-8', errors='ignore')
                    # 如果文件在父 commit 中不存在，说明是新创建的文件
                    if "does not exist in" in error_message or "exists on disk, but not in" in error_message:
                        process_log.append(f"文件 {file_path} 在 commit {commit_hash} 的父 commit 中不存在，跳过。")
                    else:
                        # 其他错误
                        process_log.append(f"处理文件 {file_path} 时发生错误: {error_message}")
                    continue
            
            # 保存 diff 信息
            os.makedirs(output_dir, exist_ok=True)
            with open(diff_output_path, 'w') as diff_file:
                diff_file.write(diff_output_decoded)
            
            processed_commits += 1
        except subprocess.CalledProcessError as e:
            process_log.append(f"处理 commit {commit_hash} 失败: {e.stderr.decode('utf-8', errors='ignore')}")
            failed_commits += 1
    
    # 生成处理结果总结
    summary = [
        f"代码库 {repository_name} 处理完成。",
        f"成功处理的 commit 数量: {processed_commits}/{total_commits}",
        f"跳过已处理的 commit 数量: {skipped_commits}",
        f"失败的 commit 数量: {failed_commits}",
        "-" * 50
    ]
    
    return "\n".join(process_log + summary)

def extract_commit_diffs_parallel():
    """
    并行从多个代码库的 is_opt_keyword.json 中提取 commit diff 信息
    """
    # 根路径
    root_path = config.root_path
    repository_base_path = os.path.join(root_path, "repository")
    
    # 获取所有代码库名称
    repository_names = [name for name in os.listdir(repository_base_path) 
                       if os.path.isdir(os.path.join(repository_base_path, name))]
    
    print(f"找到 {len(repository_names)} 个代码库待处理，设置并行数为 {MAX_PARALLEL_REPOS}")
    print("开始并行处理代码库...")
    
    # 使用进程池并行处理多个代码库
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_REPOS) as executor:
        # 提交所有任务并获取future对象
        future_to_repo = {executor.submit(process_repository, repo_name): repo_name 
                          for repo_name in repository_names}
        
        # 使用tqdm显示进度
        for future in tqdm(future_to_repo, desc="Processing repositories", unit="repository"):
            repo_name = future_to_repo[future]
            try:
                result = future.result()
                print(result)  # 打印每个代码库的处理结果
            except Exception as e:
                print(f"代码库 {repo_name} 处理时发生异常: {str(e)}")

# 程序入口
if __name__ == "__main__":
    extract_commit_diffs_parallel()