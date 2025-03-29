import json
import os
import shutil
from git import Repo
from tqdm import tqdm
import concurrent.futures
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def get_all_commits():
    """
    遍历 config.root_path/repository 中的所有子文件夹（代码库），
    为每个代码库在 config.root_path/knowledge_base 下创建对应的子文件夹，
    并获取该代码库的所有 commit 信息，存储到 knowledge_base/<repo_name>/all_commit.json 中。
    使用并行加速以及 tqdm 显示处理进度。
    
    当局部变量 skip_existing 为 True 时，若存在目标文件 all_commit.json 则跳过该仓库的处理。
    """
    # 是否跳过已经存在 all_commit.json 的仓库
    skip_existing = True

    repository_dir = os.path.join(config.root_path, "repository")
    knowledge_base_dir = os.path.join(config.root_path, "knowledge_base")
    
    # 如果 knowledge_base 文件夹不存在，则创建
    if not os.path.exists(knowledge_base_dir):
        os.makedirs(knowledge_base_dir)
    
    # 获取 repository 目录下的所有子文件夹名称（代码库名）
    repo_names = [name for name in os.listdir(repository_dir)
                  if os.path.isdir(os.path.join(repository_dir, name))]
    print("repo len: ", len(repo_names))
    
    def process_repo(repo_name):
        repo_path = os.path.join(repository_dir, repo_name)
        # 在 knowledge_base 下创建对应的子文件夹
        kb_repo_folder = os.path.join(knowledge_base_dir, repo_name)
        if not os.path.exists(kb_repo_folder):
            os.makedirs(kb_repo_folder)
        
        output_file = os.path.join(kb_repo_folder, "all_commit.json")
        
        # 若已经存在 all_commit.json 且设置了跳过标志，则跳过此仓库处理
        if skip_existing and os.path.exists(output_file):
            print(f"仓库 {repo_name}: {output_file} 已存在，跳过处理。")
            return
        
        try:
            repo = Repo(repo_path)
        except Exception as e:
            print(f"无法加载仓库 {repo_name} ，错误信息: {e}")
            return
        
        try:
            commits = list(repo.iter_commits())
            commit_list = []
            for commit in commits:
                commit_info = {
                    "hash": commit.hexsha,
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                    "message": commit.message.strip()
                }
                commit_list.append(commit_info)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(commit_list, f, indent=4)
            print(f"仓库 {repo_name}: total {len(commit_list)} commits saved to {output_file}.")
        except Exception as e:
            print(f"处理仓库 {repo_name} 时发生错误: {e}")
    
    # 使用线程池并行处理各仓库，结合 tqdm 显示进度
    with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(process_repo, repo_name) for repo_name in repo_names]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing repositories"):
            pass




def filter_one_file_commits():
    """
    遍历 config.root_path/knowledge_base 中的所有子文件夹，处理每个子文件夹中的 all_commit.json 文件。
    从输入文件中筛选出 modified_files_count 为 1 的 commit，并写入到输出文件 one_file.json。
    """
    # 是否要跳过现有的结果
    # 如果为 True 则重新运行并覆盖已存在的one_file.json文件
    # 如果为 False 则跳过已存在one_file.json的代码库
    overwrite_existing=False

    root_result_path = os.path.join(config.root_path, "knowledge_base")
    # 获取所有子文件夹名称列表
    repo_list = [repo_name for repo_name in os.listdir(root_result_path)
                 if os.path.isdir(os.path.join(root_result_path, repo_name))]
    
    # 使用 tqdm 进度条遍历每个仓库
    for repo_name in tqdm(repo_list, desc="Processing repositories", unit="repo"):
        repo_path = os.path.join(root_result_path, repo_name)
        # 输入文件路径
        input_path = os.path.join(repo_path, "all_commit.json")
        # 输出文件路径
        output_path = os.path.join(repo_path, "one_file.json")
        
        # 检查输出文件是否已存在，如果存在且不需要覆盖，则跳过
        if os.path.exists(output_path) and not overwrite_existing:
            print(f"仓库 {repo_name} 的 one_file.json 已存在，跳过处理。")
            continue
        
        if os.path.exists(input_path):
            print(f"\n正在处理仓库: {repo_name}")
            try:
                # 读取输入文件
                with open(input_path, "r", encoding="utf-8") as f:
                    commit_data = json.load(f)
                
                # 筛选出 modified_files_count 为 1 的 commit
                filtered_commits = [
                    {key: value for key, value in commit.items()}
                    for commit in commit_data
                    if commit.get("modified_files_count", "") == 1
                ]
                
                # 确保输出文件的父目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # 将筛选结果写入输出文件
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_commits, f, indent=4, ensure_ascii=False)
                
                print(f"仓库 {repo_name}: 筛选完成，结果已写入 {output_path}。")
                print(f"共找到 {len(filtered_commits)} 个符合条件的 commit。")
            except Exception as e:
                print(f"处理文件 {input_path} 时发生错误: {e}")
        else:
            print(f"仓库 {repo_name} 中未找到 all_commit.json 文件，跳过。")




def filter_c_language_commits():
    """
    遍历 config.root_path/knowledge_base 中的所有子文件夹，处理每个子文件夹中的 one_file.json 文件。
    从输入文件中筛选出被修改文件为 C/C++ 语言的 commit，并写入到输出文件 c_language.json。
    """
    # 是否要跳过现有的结果
    # 如果为 True 则重新运行并覆盖已存在的one_file.json文件
    # 如果为 False 则跳过已存在one_file.json的代码库
    overwrite_existing=False

    # 定义 C/C++ 文件扩展名
    C_CPP_EXTENSIONS = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx'}
    # 根路径
    root_result_path = os.path.join(config.root_path, "knowledge_base")
    
    # 获取所有子文件夹名称（仓库），确保是文件夹
    repo_list = [repo_name for repo_name in os.listdir(root_result_path)
                 if os.path.isdir(os.path.join(root_result_path, repo_name))]
    
    # 使用 tqdm 进度条遍历每个仓库
    for repo_name in tqdm(repo_list, desc="Processing repositories", unit="repo"):
        repo_path = os.path.join(root_result_path, repo_name)
        # 输入文件路径（one_file.json）
        input_path = os.path.join(repo_path, "one_file.json")
        # 输出文件路径
        output_path = os.path.join(repo_path, "c_language.json")
        
        # 检查输出文件是否已存在，如果存在且不需要覆盖，则跳过
        if os.path.exists(output_path) and not overwrite_existing:
            print(f"仓库 {repo_name} 的 c_language.json 已存在，跳过处理。")
            continue
        
        if os.path.exists(input_path):
            print(f"\n正在处理仓库: {repo_name}")
            try:
                # 读取输入文件
                with open(input_path, "r", encoding="utf-8") as f:
                    commit_data = json.load(f)
                    
                # 筛选出修改的文件是 C/C++ 文件的 commit
                filtered_commits = []
                for commit in commit_data:
                    modified_files = commit.get("modified_files", [])
                    # 每个 commit 只修改一个文件的假设（如果不止，可以在这里做适当调整）
                    if modified_files:
                        file_name = modified_files[0]
                        _, file_extension = os.path.splitext(file_name)
                        if file_extension.lower() in C_CPP_EXTENSIONS:
                            filtered_commits.append(commit)
                
                # 确保输出文件的父目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 将筛选结果写入输出文件
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_commits, f, indent=4, ensure_ascii=False)
                
                print(f"仓库 {repo_name}: 筛选完成，结果已写入 {output_path}。")
                print(f"共找到 {len(filtered_commits)} 个符合条件的 commit。")
            except Exception as e:
                print(f"处理文件 {input_path} 时发生错误: {e}")
        else:
            print(f"仓库 {repo_name} 中未找到 one_file.json 文件，跳过。")



def delete_modified_file_folders():
    """
    删除 root_path/result/<repository name> 中的 modified_file 文件夹。
    """
    # 根路径
    root_result_path = os.path.join(config.root_path, "result")

    # 遍历 root_path/result 中的所有子文件夹
    for repo_name in os.listdir(root_result_path):
        repo_path = os.path.join(root_result_path, repo_name)
        
        # 确保是文件夹
        if os.path.isdir(repo_path):
            # modified_file 文件夹路径
            modified_file_folder_path = os.path.join(repo_path, "modified_file")
            
            # 如果 modified_file 文件夹存在，则删除
            if os.path.exists(modified_file_folder_path):
                try:
                    shutil.rmtree(modified_file_folder_path)
                    print(f"已删除仓库 {repo_name} 中的 modified_file 文件夹。")
                except Exception as e:
                    print(f"删除仓库 {repo_name} 中的 modified_file 文件夹时发生错误: {e}")
            else:
                print(f"仓库 {repo_name} 中未找到 modified_file 文件夹，跳过。")




def get_has_file():
    """
    遍历所有代码库，过滤 is_opt_keyword.json 中所有符合条件的 commit（在 result/<repo name>/modified_file 中存在 commit hash 对应的子文件夹，
    且该文件夹中同时包含 before.txt, after.txt, diff.txt 三个文件），并将其信息保存到 has_file.json 中。
    使用并行处理提高效率，并增加进度条显示。
    """
    # 根路径
    root_path = config.root_path
    result_base_path = os.path.join(root_path, "knowledge_base")
    
    # 获取所有代码库目录
    all_repositories = [repo for repo in os.listdir(result_base_path) 
                       if os.path.isdir(os.path.join(result_base_path, repo))]
    
    print(f"开始并行处理 {len(all_repositories)} 个代码库...")
    
    # 使用线程池执行并行处理
    results = []
    
    def process_repository(repository_name):
        """处理单个代码库"""
        repository_path = os.path.join(result_base_path, repository_name)
        # 检查是否是代码库目录
        if not os.path.isdir(repository_path):
            return repository_name, 0, "不是有效目录"
        
        # 定义路径
        json_file_path = os.path.join(repository_path, "is_opt_keyword.json")
        modified_file_path = os.path.join(repository_path, "modified_file")
        final_benchmark_path = os.path.join(repository_path, "has_file.json")
        
        # 检查 is_opt_keyword.json 是否存在
        if not os.path.exists(json_file_path):
            return repository_name, 0, "is_opt_keyword.json not found"
        
        # 检查 modified_file 文件夹是否存在
        if not os.path.exists(modified_file_path):
            return repository_name, 0, "modified_file folder not found"
        
        try:
            # 读取 is_opt_keyword.json 文件
            with open(json_file_path, 'r') as f:
                commits = json.load(f)
            
            # 初始化符合条件的 commit 列表
            filtered_commits = []
            skipped_commits = []  # 用于记录不符合条件的commit
            
            # 遍历每个 commit
            for commit in commits:
                commit_hash = commit['hash']
                commit_folder = os.path.join(modified_file_path, commit_hash)
                
                # 检查是否存在对应的 commit hash 文件夹
                if not (os.path.exists(commit_folder) and os.path.isdir(commit_folder)):
                    skipped_commits.append((commit_hash, "文件夹不存在"))
                    continue
                
                # 检查三个必需文件是否都存在
                before_file = os.path.join(commit_folder, 'before.txt')
                after_file = os.path.join(commit_folder, 'after.txt')
                diff_file = os.path.join(commit_folder, 'diff.txt')
                
                if (os.path.exists(before_file) and 
                    os.path.exists(after_file) and 
                    os.path.exists(diff_file)):
                    # 所有三个文件都存在，将该 commit 添加到过滤后的列表中
                    filtered_commits.append(commit)
                else:
                    # 记录缺少的文件
                    missing_files = []
                    if not os.path.exists(before_file):
                        missing_files.append("before.txt")
                    if not os.path.exists(after_file):
                        missing_files.append("after.txt")
                    if not os.path.exists(diff_file):
                        missing_files.append("diff.txt")
                    skipped_commits.append((commit_hash, f"缺少文件: {', '.join(missing_files)}"))
            
            # 将过滤后的 commit 列表保存到 has_file.json
            with open(final_benchmark_path, 'w') as f:
                json.dump(filtered_commits, f, indent=4)
            
            # 记录被跳过的commit信息到日志文件
            if skipped_commits:
                skipped_log_path = os.path.join(repository_path, "skipped_commits.log")
                with open(skipped_log_path, 'w') as f:
                    for commit_hash, reason in skipped_commits:
                        f.write(f"{commit_hash}: {reason}\n")
            
            return repository_name, len(filtered_commits), "成功", len(skipped_commits)
        except Exception as e:
            return repository_name, 0, f"错误: {str(e)}", 0
    
    # 使用线程池执行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        # 提交所有任务
        future_to_repo = {executor.submit(process_repository, repo): repo for repo in all_repositories}
        
        # 使用tqdm显示进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_repo), 
                          total=len(all_repositories), 
                          desc="处理代码库进度"):
            repo = future_to_repo[future]
            try:
                result = future.result()
                if len(result) == 4:
                    repo_name, commit_count, status, skipped_count = result
                    results.append((repo_name, commit_count, status, skipped_count))
                else:
                    repo_name, commit_count, status = result
                    results.append((repo_name, commit_count, status, 0))
            except Exception as e:
                print(f"处理代码库 {repo} 出错: {e}")
                results.append((repo, 0, f"错误: {str(e)}", 0))
    
    # 打印结果统计
    print("-" * 80)
    print("处理完成! 结果统计:")
    success_count = sum(1 for _, count, status, _ in results if status == "成功" and count > 0)
    total_commits = sum(count for _, count, status, _ in results if status == "成功")
    total_skipped = sum(skipped for _, _, status, skipped in results if status == "成功")
    
    print(f"成功处理的代码库: {success_count}/{len(all_repositories)}")
    print(f"总共过滤的有效commit数: {total_commits}")
    print(f"总共跳过的不符合条件的commit数: {total_skipped}")
    
    # 详细输出每个代码库的处理结果
    for repo_name, commit_count, status, skipped_count in sorted(results, key=lambda x: x[1], reverse=True):
        if status == "成功":
            if skipped_count > 0:
                print(f"{repo_name}: 过滤得到 {commit_count} 个符合条件的commit, 跳过 {skipped_count} 个不符合条件的commit")
            else:
                print(f"{repo_name}: 过滤得到 {commit_count} 个符合条件的commit")
        else:
            print(f"{repo_name}: {status}")
    
    print("-" * 80)
    return results




def copy_has_file_to_func():
    """
    遍历 config.root_path/knowledge_base 中的所有代码库，对于每个代码库，
    将 has_file.json 文件复制到 has_file_with_func.json 文件中。如果目标文件已存在则直接覆盖。
    """
    knowledge_base_path = os.path.join(config.root_path, "knowledge_base")
    
    # 检查 knowledge_base 目录是否存在
    if not os.path.exists(knowledge_base_path):
        print(f"目录 {knowledge_base_path} 不存在")
        return
    
    for repo_name in os.listdir(knowledge_base_path):
        repo_path = os.path.join(knowledge_base_path, repo_name)
        # 仅处理目录（代码库）
        if os.path.isdir(repo_path):
            src_file = os.path.join(repo_path, "has_file.json")
            dst_file = os.path.join(repo_path, "has_file_with_func.json")
            
            # 检查源文件是否存在
            if os.path.exists(src_file):
                try:
                    # 复制文件，shutil.copyfile 会覆盖已经存在的目标文件
                    shutil.copyfile(src_file, dst_file)
                    print(f"成功复制文件:\n  {src_file}\n到\n  {dst_file}")
                except Exception as e:
                    print(f"复制 {src_file} 到 {dst_file} 时出错: {e}")
            else:
                print(f"文件 {src_file} 不存在，跳过仓库：{repo_name}")




def filter_one_func_commits():
    """
    遍历 config.root_path/knowledge_base 中的所有子文件夹，处理每个子文件夹中的 has_file_with_func.json 文件。
    筛选条件：commit对应的modified_file目录下同时存在before_func.txt和after_func.txt文件。
    将筛选结果覆盖写入 one_func.json 中。
    """
    max_workers = 128
    
    # 获取知识库根路径
    root_result_path = os.path.join(config.root_path, "knowledge_base")
    
    # 获取仓库列表
    repositories = [repo_name for repo_name in os.listdir(root_result_path) 
                   if os.path.isdir(os.path.join(root_result_path, repo_name))]
    
    print(f"找到 {len(repositories)} 个代码库，开始筛选单函数修改提交...")
    print(f"使用 {max_workers} 个并行线程进行处理")
    
    # 处理单个仓库的内部函数
    def process_repo(repo_name):
        repo_path = os.path.join(root_result_path, repo_name)
        
        # 输入文件路径
        input_path = os.path.join(repo_path, "has_file_with_func.json")
        
        # 输出文件路径
        output_path = os.path.join(repo_path, "one_func.json")
        
        # 如果输入文件存在，则处理
        if os.path.exists(input_path):
            try:
                # 读取输入文件
                with open(input_path, "r", encoding="utf-8") as f:
                    commit_data = json.load(f)
                
                # 筛选出对应目录下同时存在before_func.txt和after_func.txt的commit
                filtered_commits = []
                for commit in commit_data:
                    commit_hash = commit.get("hash")
                    if not commit_hash:
                        continue
                    
                    # 构建文件路径
                    modified_file_dir = os.path.join(repo_path, "modified_file", commit_hash)
                    before_func_path = os.path.join(modified_file_dir, "before_func.txt")
                    after_func_path = os.path.join(modified_file_dir, "after_func.txt")
                    
                    # 检查文件是否存在
                    if os.path.isfile(before_func_path) and os.path.isfile(after_func_path):
                        filtered_commits.append(commit)
                
                # 确保输出文件的父目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 将筛选结果写入输出文件
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_commits, f, indent=4, ensure_ascii=False)
                
                return repo_name, len(filtered_commits), True
            except Exception as e:
                return repo_name, 0, f"错误: {e}"
        else:
            return repo_name, 0, "未找到 has_file_with_func.json 文件"
    
    # 初始化统计信息
    total_success = 0
    total_commits = 0
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有仓库处理任务
        future_to_repo = {executor.submit(process_repo, repo_name): repo_name 
                         for repo_name in repositories}
        
        # 创建进度条
        with tqdm(total=len(repositories), desc="处理仓库") as pbar:
            for future in concurrent.futures.as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    repo_name, commit_count, result = future.result()
                    
                    # 更新进度条
                    pbar.update(1)
                    
                    # 处理结果
                    if result is True:
                        total_success += 1
                        total_commits += commit_count
                        if commit_count > 0:
                            pbar.set_postfix_str(f"当前: {repo_name} ({commit_count}个提交)")
                    else:
                        tqdm.write(f"[{repo_name}] {result}")
                except Exception as e:
                    pbar.update(1)
                    tqdm.write(f"[{repo_name}] 处理异常: {str(e)}")
    
    print(f"\n筛选完成! 成功处理 {total_success} 个代码库，找到 {total_commits} 个单函数修改提交。")



def aggregate_and_deduplicate_one_func():
    """
    遍历所有代码库中的 one_func.json，为每个 commit 添加 repository_name 字段，
    将所有 commit 汇总到 root_path/all_one_func.json 中，进行去重，
    然后将结果按代码库分发到各自的 one_func_deduplicate.json 文件中。
    """
    # 根路径
    root_path = config.root_path
    knowledge_base_path = os.path.join(root_path, "knowledge_base")
    
    # 初始化所有代码库的 commit 列表
    all_commits = []
    repository_commit_counts = {}
    
    # 遍历所有代码库
    print("第一阶段：收集所有代码库的 one_func.json 文件...")
    for repository_name in tqdm(os.listdir(knowledge_base_path)):
        repository_path = os.path.join(knowledge_base_path, repository_name)
        
        # 检查是否是代码库目录
        if not os.path.isdir(repository_path):
            continue
            
        # 定义 one_func.json 路径
        one_func_path = os.path.join(repository_path, "one_func.json")
        
        # 检查 one_func.json 是否存在
        if not os.path.exists(one_func_path):
            continue
            
        try:
            # 读取 one_func.json 文件
            with open(one_func_path, 'r', encoding='utf-8') as f:
                commits = json.load(f)
                
            # 为每个 commit 添加 repository_name 字段，并放在第一个字段
            repo_commits = []
            for commit in commits:
                commit_with_repo = {"repository_name": repository_name}
                commit_with_repo.update(commit)
                repo_commits.append(commit_with_repo)
                
            # 记录原始数量
            repository_commit_counts[repository_name] = len(repo_commits)
            
            # 添加到全局列表
            all_commits.extend(repo_commits)
            
        except Exception as e:
            print(f"处理代码库 {repository_name} 的 one_func.json 时出错: {e}")
    
    # 打印收集信息
    print(f"成功收集 {len(repository_commit_counts)} 个代码库的 commit，共 {len(all_commits)} 个 commit。")
    
    # 保存汇总文件
    all_one_func_path = os.path.join(root_path, "all_one_func.json")
    with open(all_one_func_path, 'w', encoding='utf-8') as f:
        json.dump(all_commits, f, indent=4, ensure_ascii=False)
    print(f"已将所有 commit 保存到 {all_one_func_path}")
    
    # 对 commit 进行去重
    print("\n第二阶段：对所有 commit 进行去重...")
    
    # 创建 hash+message 到 commit 的映射字典，用于检测重复
    unique_commits_dict = {}
    
    for commit in tqdm(all_commits):
        # 使用 commit hash 和 message 作为去重标识
        dedup_key = (commit.get('hash', ''), commit.get('message', ''))
        
        # 如果是新的 commit，加入字典
        if dedup_key not in unique_commits_dict:
            unique_commits_dict[dedup_key] = commit
    
    # 转换回列表形式
    unique_commits = list(unique_commits_dict.values())
    
    # 打印去重信息
    duplicate_count = len(all_commits) - len(unique_commits)
    print(f"去重完成，共删除了 {duplicate_count} 个重复 commit，剩余 {len(unique_commits)} 个唯一 commit。")
    
    # 保存去重后的文件
    dedup_file_path = os.path.join(root_path, "all_one_func_deduplicated.json")
    with open(dedup_file_path, 'w', encoding='utf-8') as f:
        json.dump(unique_commits, f, indent=4, ensure_ascii=False)
    print(f"已将去重后的 commit 保存到 {dedup_file_path}")
    
    # 将去重后的结果分发回各个代码库
    print("\n第三阶段：将去重后的结果分发到各代码库...")
    
    # 按代码库分组
    commits_by_repo = {}
    for commit in unique_commits:
        repo_name = commit.get('repository_name')
        if repo_name:
            if repo_name not in commits_by_repo:
                commits_by_repo[repo_name] = []
            commits_by_repo[repo_name].append(commit)
    
    # 分发到各个代码库
    for repo_name, repo_commits in tqdm(commits_by_repo.items()):
        output_path = os.path.join(knowledge_base_path, repo_name, "one_func_deduplicate.json")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存到对应的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(repo_commits, f, indent=4, ensure_ascii=False)
        
        # 计算去重率
        original_count = repository_commit_counts.get(repo_name, 0)
        dedup_count = len(repo_commits)
        if original_count > 0:
            dedup_rate = (original_count - dedup_count) / original_count * 100
        else:
            dedup_rate = 0
            
        print(f"代码库 {repo_name}: 原始 {original_count} 个 commit，去重后 {dedup_count} 个，去重率 {dedup_rate:.2f}%")
    
    print("\n全部处理完成！")
    
    return {
        "total_original": len(all_commits),
        "total_deduplicated": len(unique_commits),
        "deduplication_rate": (len(all_commits) - len(unique_commits)) / len(all_commits) * 100 if all_commits else 0
    }



def filter_optimization_commits():
    """
    遍历 root_path/knowledge_base 中的所有子文件夹，处理每个子文件夹中的 is_opt_llm.json 文件。
    从输入文件中筛选出 is_opt_ds_simple 为 true 的 commit，并写入到输出文件 is_opt_final.json。
    使用并行处理提高效率，并增加进度条显示。
    """
    root_result_path = os.path.join(config.root_path, "knowledge_base")
    
    # 获取所有仓库目录
    repositories = [repo_name for repo_name in os.listdir(root_result_path) 
                   if os.path.isdir(os.path.join(root_result_path, repo_name))]
    
    print(f"开始并行处理 {len(repositories)} 个代码库...")
    
    # 定义处理单个仓库的函数
    def process_repository(repo_name):
        repo_path = os.path.join(root_result_path, repo_name)
        
        # 输入文件路径
        input_path = os.path.join(repo_path, "is_opt_llm.json")
        
        # 输出文件路径
        output_path = os.path.join(repo_path, "is_opt_final.json")
        
        # 如果输入文件存在，则处理
        if os.path.exists(input_path):
            try:
                # 读取输入文件
                with open(input_path, "r") as f:
                    commit_data = json.load(f)
                
                # 筛选出 is_opt_ds_simple 为 true 的 commit，保留该字段
                filtered_commits = [
                    commit for commit in commit_data
                    if commit.get("is_opt_ds_simple", "").lower() == "true"
                ]
                
                # 确保输出文件的父目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 将筛选结果写入输出文件
                with open(output_path, "w") as f:
                    json.dump(filtered_commits, f, indent=4)
                
                return repo_name, len(filtered_commits), "成功"
            except Exception as e:
                return repo_name, 0, f"错误: {str(e)}"
        else:
            return repo_name, 0, "未找到 is_opt_llm.json 文件"
    
    # 设置线程池的最大线程数为8
    max_workers = 8
    results = []
    
    # 使用线程池执行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_repo = {executor.submit(process_repository, repo): repo for repo in repositories}
        
        # 使用tqdm显示进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_repo), 
                          total=len(repositories), 
                          desc="处理仓库进度"):
            repo = future_to_repo[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"处理仓库 {repo} 出错: {e}")
                results.append((repo, 0, f"错误: {str(e)}"))
    
    # 打印结果统计
    print("-" * 80)
    print("处理完成! 结果统计:")
    success_count = sum(1 for _, count, status in results if status == "成功")
    total_commits = sum(count for _, count, status in results if status == "成功")
    
    print(f"成功处理的仓库: {success_count}/{len(repositories)}")
    print(f"总共找到的符合条件的commit数: {total_commits}")
    
    # 详细输出每个仓库的处理结果
    for repo_name, commit_count, status in sorted(results, key=lambda x: x[1], reverse=True):
        if status == "成功":
            print(f"{repo_name}: 找到 {commit_count} 个符合条件的commit")
        else:
            print(f"{repo_name}: {status}")
    
    print("-" * 80)
    return results



def aggregate_final_commits():
    """
    遍历 root_path/knowledge_base 中的所有子文件夹，读取每个子文件夹中的 is_opt_final.json 文件。
    将所有符合条件的 commit 汇总到 root_path/all_is_opt_final.json 文件中，并删除每个 commit 中的 all_functions 字段。
    同时每个 commit 添加字段 repository_name，记录该 commit 所在的仓库名称。
    """
    # 定义结果路径
    root_result_path = os.path.join(config.root_path, "knowledge_base")
    output_path = os.path.join(config.root_path, "all_is_opt_final.json")
    aggregated_commits = []

    # 遍历 root_path/result 中的所有子文件夹
    for repo_name in os.listdir(root_result_path):
        repo_path = os.path.join(root_result_path, repo_name)
        
        # 确保是文件夹
        if os.path.isdir(repo_path):
            # 输入文件路径
            input_path = os.path.join(repo_path, "is_opt_final.json")
            
            # 如果输入文件存在，则处理
            if os.path.exists(input_path):
                print(f"正在处理仓库: {repo_name}")
                try:
                    # 读取输入文件
                    with open(input_path, "r", encoding="utf-8") as f:
                        commit_data = json.load(f)
                    
                    # 处理每个 commit
                    for commit in commit_data:
                        # 删除 all_functions 字段
                        if 'all_functions' in commit:
                            del commit['all_functions']
                        # 添加 repository_name 字段
                        commit['repository_name'] = repo_name
                        # 将处理后的 commit 添加到汇总列表中
                        aggregated_commits.append(commit)
                    
                    print(f"已添加 {len(commit_data)} 个 commit 来自仓库 {repo_name}。")
                except Exception as e:
                    print(f"处理文件 {input_path} 时发生错误: {e}")
            else:
                print(f"仓库 {repo_name} 中未找到 one_func.json 文件，跳过。")

    # 写入汇总后的 commit 数据
    try:
        # 确保输出文件的父目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_commits, f, indent=4, ensure_ascii=False)
        
        print(f"\n所有符合条件的 commit 已汇总到 {output_path}。\n")
        print(f"共汇总了 {len(aggregated_commits)} 个符合条件的 commit。")
    except Exception as e:
        print(f"写入文件 {output_path} 时发生错误: {e}")



# get_all_commits()

# filter_one_file_commits()
# filter_c_language_commits()

# get_has_file()
# copy_has_file_to_func()

# aggregate_and_deduplicate_one_func()

# filter_one_func_commits()

# filter_optimization_commits()

# aggregate_final_commits()
