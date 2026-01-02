import json
import os
import shutil
import glob
from git import Repo
from tqdm import tqdm
import concurrent.futures
import re
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
    # repo_names = ["fcpw", "pi-defender", "VapourSynth-Editor", "Capturinha"]
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
    overwrite_existing=True

    root_result_path = os.path.join(config.root_path, "knowledge_base")
    # 获取所有子文件夹名称列表
    repo_list = [repo_name for repo_name in os.listdir(root_result_path)
                 if os.path.isdir(os.path.join(root_result_path, repo_name))]
    # repo_list = ["fcpw", "pi-defender", "VapourSynth-Editor", "Capturinha"]
    
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
    且该文件夹中同时包含唯一的 before.*、唯一的 after.*、和 diff.txt 三个文件），并将其信息保存到 has_file.json 中。
    使用并行处理提高效率，并增加进度条显示。
    """
    # 根路径
    root_path = config.root_path
    result_base_path = os.path.join(root_path, "knowledge_base_all")
    
    # 获取所有代码库目录
    all_repositories = [
        repo
        for repo in os.listdir(result_base_path)
        if os.path.isdir(os.path.join(result_base_path, repo))
    ]
    
    print(f"开始并行处理 {len(all_repositories)} 个代码库...")
    
    results = []
    
    def process_repository(repository_name):
        """处理单个代码库"""
        repository_path = os.path.join(result_base_path, repository_name)
        if not os.path.isdir(repository_path):
            return repository_name, 0, "不是有效目录"
        
        json_file_path = os.path.join(repository_path, "is_opt_keyword.json")
        modified_file_path = os.path.join(repository_path, "modified_file")
        final_benchmark_path = os.path.join(repository_path, "has_file.json")
        
        if not os.path.exists(json_file_path):
            return repository_name, 0, "is_opt_keyword.json not found"
        if not os.path.exists(modified_file_path):
            return repository_name, 0, "modified_file folder not found"
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                commits = json.load(f)
            
            filtered_commits = []
            skipped_commits  = []
            
            for commit in commits:
                commit_hash   = commit['hash']
                commit_folder = os.path.join(modified_file_path, commit_hash)
                
                if not (os.path.exists(commit_folder) and os.path.isdir(commit_folder)):
                    skipped_commits.append((commit_hash, "文件夹不存在"))
                    continue
                
                # 1) 匹配 before.*
                all_files = os.listdir(commit_folder)
                before_matches = [f for f in all_files if re.match(r"^before\..+$", f)]
                if len(before_matches) != 1:
                    reason = (
                        "未找到唯一的 before.* 文件"
                        if len(before_matches) == 0
                        else f"匹配到多个 before.* 文件: {before_matches}"
                    )
                    skipped_commits.append((commit_hash, reason))
                    continue
                before_file = os.path.join(commit_folder, before_matches[0])
                
                # 2) 匹配 after.*
                after_matches = [f for f in all_files if re.match(r"^after\..+$", f)]
                if len(after_matches) != 1:
                    reason = (
                        "未找到唯一的 after.* 文件"
                        if len(after_matches) == 0
                        else f"匹配到多个 after.* 文件: {after_matches}"
                    )
                    skipped_commits.append((commit_hash, reason))
                    continue
                after_file = os.path.join(commit_folder, after_matches[0])
                
                # 3) 检查 diff.txt
                diff_file = os.path.join(commit_folder, 'diff.txt')
                if not os.path.exists(diff_file):
                    skipped_commits.append((commit_hash, "缺少 diff.txt"))
                    continue
                
                # 如果都 OK，则加入 filtered_commits
                filtered_commits.append(commit)
            
            # 保存 has_file.json
            with open(final_benchmark_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_commits, f, indent=4, ensure_ascii=False)
            
            # 保存 skipped_commits.log
            if skipped_commits:
                skipped_log_path = os.path.join(repository_path, "skipped_commits.log")
                with open(skipped_log_path, 'w', encoding='utf-8') as f:
                    for ch, reason in skipped_commits:
                        f.write(f"{ch}: {reason}\n")
            
            return repository_name, len(filtered_commits), "成功", len(skipped_commits)
        
        except Exception as e:
            return repository_name, 0, f"错误: {str(e)}", 0
    
    # 并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
        future_to_repo = {
            executor.submit(process_repository, repo): repo
            for repo in all_repositories
        }
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_repo),
            total=len(all_repositories),
            desc="处理代码库进度"
        ):
            repo = future_to_repo[future]
            try:
                res = future.result()
                # 统一四元组格式
                if len(res) == 4:
                    results.append(res)
                else:
                    repo_name, c, status = res
                    results.append((repo_name, c, status, 0))
            except Exception as e:
                print(f"处理代码库 {repo} 出错: {e}")
                results.append((repo, 0, f"错误: {str(e)}", 0))
    
    # 汇总输出
    print("-" * 80)
    print("处理完成! 结果统计:")
    success_count = sum(1 for _, cnt, st, _ in results if st == "成功")
    total_commits = sum(cnt for _, cnt, st, _ in results if st == "成功")
    total_skipped = sum(sk for _, _, st, sk in results if st == "成功")
    
    print(f"成功处理的代码库: {success_count}/{len(all_repositories)}")
    print(f"总共过滤的有效commit数: {total_commits}")
    print(f"总共跳过的不符合条件的commit数: {total_skipped}")
    
    for repo_name, cnt, status, sk in sorted(results, key=lambda x: x[1], reverse=True):
        if status == "成功":
            if sk > 0:
                print(f"{repo_name}: 过滤得到 {cnt} 个符合条件的commit, 跳过 {sk} 个不符合条件的commit")
            else:
                print(f"{repo_name}: 过滤得到 {cnt} 个符合条件的commit")
        else:
            print(f"{repo_name}: {status}")
    
    print("-" * 80)
    return results



def copy_has_file():
    """
    遍历 config.root_path/knowledge_base 中的所有代码库，对于每个代码库，
    将 has_file_deduplicate.json 文件复制到 diff.json 文件中。如果目标文件已存在则直接覆盖。
    """
    knowledge_base_path = os.path.join(config.root_path, "knowledge_base_all")
    
    # 检查 knowledge_base 目录是否存在
    if not os.path.exists(knowledge_base_path):
        print(f"目录 {knowledge_base_path} 不存在")
        return
    
    for repo_name in os.listdir(knowledge_base_path):
        repo_path = os.path.join(knowledge_base_path, repo_name)
        # 仅处理目录（代码库）
        if os.path.isdir(repo_path):
            src_file = os.path.join(repo_path, "has_file_deduplicate.json")
            dst_file = os.path.join(repo_path, "diff.json")
            
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
    遍历 root_path/knowledge_base 中的所有子文件夹，处理每个子文件夹中的 diff.json 文件。
    从输入文件中筛选出 modified_func_count = 1 且 modified_other 为 false 且存在唯一 before_func.* 文件
    且存在唯一 after_func.* 文件的 commit，并写入到输出文件 one_func.json。
    使用并行处理提高效率，并增加进度条显示。
    """
    root_result_path = os.path.join(config.root_path, "knowledge_base_all")
    # 获取所有仓库目录
    repositories = [repo_name for repo_name in os.listdir(root_result_path)
                   if os.path.isdir(os.path.join(root_result_path, repo_name))]
    print(f"开始并行处理 {len(repositories)} 个代码库...")
    
    # 定义处理单个仓库的函数
    def process_repository(repo_name):
        repo_path = os.path.join(root_result_path, repo_name)
        # 输入文件路径
        input_path = os.path.join(repo_path, "diff.json")
        # 输出文件路径
        output_path = os.path.join(repo_path, "one_func.json")
        
        # 如果输入文件存在，则处理
        if os.path.exists(input_path):
            try:
                # 读取输入文件
                with open(input_path, "r") as f:
                    commit_data = json.load(f)
                
                # 第一步筛选：modified_func_count = 1 且 modified_other 为 false
                initial_filtered_commits = [
                    commit for commit in commit_data
                    if commit.get("modified_func_count", 0) == 1 and 
                       commit.get("modified_other", True) == False
                ]
                
                # 第二步筛选：检查 before_func.* 和 after_func.* 文件
                final_filtered_commits = []
                for commit in initial_filtered_commits:
                    commit_hash = commit.get('hash', '')
                    if not commit_hash:
                        continue
                    
                    # 构建 commit 文件夹路径
                    commit_dir = os.path.join(
                        root_result_path,
                        repo_name,
                        "modified_file",
                        commit_hash
                    )
                    
                    # 检查 commit 文件夹是否存在
                    if not os.path.exists(commit_dir):
                        continue
                    
                    # 使用 glob 查找 before_func.* 文件
                    before_func_pattern = os.path.join(commit_dir, "before_func.*")
                    before_func_files = glob.glob(before_func_pattern)
                    
                    # 使用 glob 查找 after_func.* 文件
                    after_func_pattern = os.path.join(commit_dir, "after_func.*")
                    after_func_files = glob.glob(after_func_pattern)
                    
                    # 只有当存在唯一的 before_func.* 文件和唯一的 after_func.* 文件时才符合条件
                    if len(before_func_files) == 1 and len(after_func_files) == 1:
                        final_filtered_commits.append(commit)
                
                # 确保输出文件的父目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 将筛选结果写入输出文件
                with open(output_path, "w") as f:
                    json.dump(final_filtered_commits, f, indent=4)
                
                return repo_name, len(initial_filtered_commits), len(final_filtered_commits), "成功"
            except Exception as e:
                return repo_name, 0, 0, f"错误: {str(e)}"
        else:
            return repo_name, 0, 0, "未找到 diff.json 文件"
    
    # 设置线程池的最大线程数为8
    max_workers = 128
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
                results.append((repo, 0, 0, f"错误: {str(e)}"))
    
    # 打印结果统计
    print("-" * 80)
    print("处理完成! 结果统计:")
    success_count = sum(1 for _, _, _, status in results if status == "成功")
    total_initial_commits = sum(initial_count for _, initial_count, _, status in results if status == "成功")
    total_final_commits = sum(final_count for _, _, final_count, status in results if status == "成功")
    
    print(f"成功处理的仓库: {success_count}/{len(repositories)}")
    print(f"初步筛选出的commit数（满足基本条件）: {total_initial_commits}")
    print(f"最终符合条件的commit数（包含唯一before_func和after_func文件）: {total_final_commits}")
    if total_initial_commits > 0:
        print(f"通过before_func和after_func文件检查的比例: {total_final_commits/total_initial_commits*100:.2f}%")
    
    # 详细输出每个仓库的处理结果
    print("\n各仓库详细结果:")
    for repo_name, initial_count, final_count, status in sorted(results, key=lambda x: x[2], reverse=True):
        if status == "成功":
            print(f"{repo_name}: 初步筛选 {initial_count} 个 -> 最终符合 {final_count} 个commit")
        else:
            print(f"{repo_name}: {status}")
    
    print("-" * 80)
    return results



def copy_one_func_file():
    """
    遍历 config.root_path/knowledge_base 中的所有代码库，对于每个代码库，
    将 one_func.json 文件复制到 line_block.json 文件中。如果目标文件已存在则直接覆盖。
    """
    knowledge_base_path = os.path.join(config.root_path, "knowledge_base_all")
    
    # 检查 knowledge_base 目录是否存在
    if not os.path.exists(knowledge_base_path):
        print(f"目录 {knowledge_base_path} 不存在")
        return
    
    for repo_name in os.listdir(knowledge_base_path):
        repo_path = os.path.join(knowledge_base_path, repo_name)
        # 仅处理目录（代码库）
        if os.path.isdir(repo_path):
            src_file = os.path.join(repo_path, "one_func.json")
            dst_file = os.path.join(repo_path, "line_block.json")
            
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




def copy_func_name_file(merge_mode=False):
    """
    遍历 config.root_path/knowledge_base 中的所有代码库，对于每个代码库，
    将 func_name_result.json 文件复制到 is_opt_llm.json 文件中。
    
    参数:
    merge_mode (bool): 如果为 False，直接覆盖目标文件；如果为 True，智能合并两个文件的commit信息
                      - 对于同时存在于源文件和目标文件的commit，使用目标文件中的信息
                      - 对于只存在于源文件的commit，复制到目标文件
                      - 对于只存在于目标文件的commit，从目标文件中删除
    """
    knowledge_base_path = os.path.join(config.root_path, "knowledge_base_all")
    
    # 检查 knowledge_base 目录是否存在
    if not os.path.exists(knowledge_base_path):
        print(f"目录 {knowledge_base_path} 不存在")
        return
    
    for repo_name in os.listdir(knowledge_base_path):
        repo_path = os.path.join(knowledge_base_path, repo_name)
        # 仅处理目录（代码库）
        if os.path.isdir(repo_path):
            src_file = os.path.join(repo_path, "func_name_result.json")
            dst_file = os.path.join(repo_path, "is_opt_llm.json")
            
            # 检查源文件是否存在
            if os.path.exists(src_file):
                try:
                    if not merge_mode:
                        # 直接覆盖模式
                        shutil.copyfile(src_file, dst_file)
                        print(f"成功复制文件:\n  {src_file}\n到\n  {dst_file}")
                    else:
                        # 智能合并模式
                        # 读取源文件
                        with open(src_file, "r", encoding="utf-8") as f:
                            src_commits = json.load(f)
                        
                        # 读取目标文件（如果存在）
                        dst_commits = []
                        if os.path.exists(dst_file):
                            with open(dst_file, "r", encoding="utf-8") as f:
                                dst_commits = json.load(f)
                        
                        # 创建commit标识函数
                        def get_commit_key(commit):
                            return (commit.get("repository_name", ""), commit.get("hash", ""))
                        
                        # 将源文件和目标文件的commit转换为字典，便于查找
                        src_dict = {get_commit_key(commit): commit for commit in src_commits}
                        dst_dict = {get_commit_key(commit): commit for commit in dst_commits}
                        
                        # 获取所有commit的键
                        src_keys = set(src_dict.keys())
                        dst_keys = set(dst_dict.keys())
                        
                        # 统计信息
                        common_keys = src_keys & dst_keys  # 同时存在于源文件和目标文件
                        src_only_keys = src_keys - dst_keys  # 只存在于源文件
                        dst_only_keys = dst_keys - src_keys  # 只存在于目标文件
                        
                        # 构建最终结果
                        final_commits = []
                        
                        # 1. 对于同时存在的commit，使用目标文件中的信息
                        for key in common_keys:
                            final_commits.append(dst_dict[key])
                        
                        # 2. 对于只存在于源文件的commit，添加到结果中
                        for key in src_only_keys:
                            final_commits.append(src_dict[key])
                        
                        # 3. 只存在于目标文件的commit被忽略（相当于删除）
                        
                        # 按hash排序，保持一致性
                        final_commits.sort(key=lambda x: x.get("hash", ""))
                        
                        # 写入目标文件
                        with open(dst_file, "w", encoding="utf-8") as f:
                            json.dump(final_commits, f, indent=4, ensure_ascii=False)
                        
                        # 输出统计信息
                        result_info = (f"仓库 {repo_name} 合并完成: "
                                      f"保留目标文件中的commit: {len(common_keys)}个, "
                                      f"从源文件添加commit: {len(src_only_keys)}个, "
                                      f"从目标文件删除commit: {len(dst_only_keys)}个, "
                                      f"最终commit总数: {len(final_commits)}个")
                        print(result_info)
                        
                except Exception as e:
                    print(f"处理 {src_file} 到 {dst_file} 时出错: {e}")
            else:
                print(f"文件 {src_file} 不存在，跳过仓库：{repo_name}")




def aggregate_and_deduplicate_has_file():
    """
    遍历所有代码库中的 has_file.json，为每个 commit 添加 repository_name 字段，
    将所有 commit 汇总到 root_path/all_has_file.json 中，进行去重，
    然后将结果按代码库分发到各自的 has_file_deduplicate.json 文件中。
    
    按照 repo_list_30342.json 中的顺序处理代码库，去重时优先保留顺序靠前的commit。
    去重条件：hash、message、modified_files 三个字段都必须完全相同
    """
    # 根路径
    root_path = config.root_path
    knowledge_base_path = os.path.join(root_path, "knowledge_base_all")
    repo_list_path = os.path.join(root_path, "repo_list_30342.json")
    
    # 读取代码库列表文件
    print("读取代码库列表文件...")
    try:
        with open(repo_list_path, 'r', encoding='utf-8') as f:
            repo_list = json.load(f)
        print(f"成功读取代码库列表，共 {len(repo_list)} 个代码库")
    except Exception as e:
        print(f"读取代码库列表文件失败: {e}")
        return None
    
    # 创建代码库名称到顺序的映射，用于后续排序
    repo_order_map = {repo["name"]: idx for idx, repo in enumerate(repo_list)}
    ordered_repo_names = [repo["name"] for repo in repo_list]
    
    # 初始化所有代码库的 commit 列表
    all_commits = []
    repository_commit_counts = {}
    processed_repos = []
    
    # 按照 repo_list 中的顺序遍历代码库
    print("第一阶段：按指定顺序收集所有代码库的 has_file.json 文件...")
    for repository_name in tqdm(ordered_repo_names, desc="处理代码库"):
        repository_path = os.path.join(knowledge_base_path, repository_name)
        
        # 检查是否是代码库目录
        if not os.path.isdir(repository_path):
            print(f"警告：代码库目录不存在: {repository_name}")
            continue
            
        # 定义 has_file.json 路径
        file_path = os.path.join(repository_path, "has_file.json")
        
        # 检查 has_file.json 是否存在
        if not os.path.exists(file_path):
            print(f"警告：has_file.json 不存在: {repository_name}")
            continue
            
        try:
            # 读取 has_file.json 文件
            with open(file_path, 'r', encoding='utf-8') as f:
                commits = json.load(f)
                
            # 为每个 commit 添加 repository_name 字段和处理顺序信息
            repo_commits = []
            for commit_idx, commit in enumerate(commits):
                commit_with_repo = {"repository_name": repository_name}
                commit_with_repo.update(commit)
                # 添加全局顺序信息，用于去重时的优先级判断
                commit_with_repo["_processing_order"] = len(all_commits) + commit_idx
                repo_commits.append(commit_with_repo)
                
            # 记录原始数量
            repository_commit_counts[repository_name] = len(repo_commits)
            processed_repos.append(repository_name)
            
            # 添加到全局列表（按顺序添加）
            all_commits.extend(repo_commits)
            
        except Exception as e:
            print(f"处理代码库 {repository_name} 的 has_file.json 时出错: {e}")
    
    # 打印收集信息
    print(f"成功收集 {len(processed_repos)} 个代码库的 commit，共 {len(all_commits)} 个 commit。")
    print(f"处理顺序: {', '.join(processed_repos[:5])}{'...' if len(processed_repos) > 5 else ''}")
    
    # 保存汇总文件
    all_file_path = os.path.join(root_path, "all_has_file.json")
    # 移除临时的处理顺序字段后保存
    commits_for_save = []
    for commit in all_commits:
        commit_copy = commit.copy()
        commit_copy.pop("_processing_order", None)
        commits_for_save.append(commit_copy)
    
    with open(all_file_path, 'w', encoding='utf-8') as f:
        json.dump(commits_for_save, f, indent=4, ensure_ascii=False)
    print(f"已将所有 commit 保存到 {all_file_path}")
    
    # 对 commit 进行去重
    print("\n第二阶段：对所有 commit 进行去重（优先保留顺序靠前的commit）...")
    
    # 创建 hash+message+modified_files 到 commit 的映射字典，用于检测重复
    unique_commits_dict = {}
    duplicate_stats = {"total_duplicates": 0, "by_repo": {}}
    
    for commit in tqdm(all_commits, desc="去重处理"):
        # 获取 modified_files 列表并排序，确保顺序一致性
        modified_files = commit.get('modified_files', [])
        if isinstance(modified_files, list):
            # 对文件列表进行排序，确保相同文件集合的顺序一致
            sorted_modified_files = tuple(sorted(modified_files))
        else:
            sorted_modified_files = tuple()
        
        # 使用 commit hash、message 和 modified_files 作为去重标识
        dedup_key = (
            commit.get('hash', ''), 
            commit.get('message', ''),
            sorted_modified_files
        )
        
        # 如果是新的 commit，加入字典
        if dedup_key not in unique_commits_dict:
            unique_commits_dict[dedup_key] = commit
        else:
            # 发现重复commit，比较处理顺序，保留顺序靠前的
            existing_commit = unique_commits_dict[dedup_key]
            current_order = commit.get("_processing_order", float('inf'))
            existing_order = existing_commit.get("_processing_order", float('inf'))
            
            if current_order < existing_order:
                # 当前commit顺序更靠前，替换已存在的commit
                duplicate_stats["total_duplicates"] += 1
                removed_repo = existing_commit.get('repository_name', 'unknown')
                duplicate_stats["by_repo"][removed_repo] = duplicate_stats["by_repo"].get(removed_repo, 0) + 1
                unique_commits_dict[dedup_key] = commit
            else:
                # 保留原有commit，丢弃当前commit
                duplicate_stats["total_duplicates"] += 1
                removed_repo = commit.get('repository_name', 'unknown')
                duplicate_stats["by_repo"][removed_repo] = duplicate_stats["by_repo"].get(removed_repo, 0) + 1
    
    # 转换回列表形式，并移除临时的处理顺序字段
    unique_commits = []
    for commit in unique_commits_dict.values():
        commit_copy = commit.copy()
        commit_copy.pop("_processing_order", None)
        unique_commits.append(commit_copy)
    
    # 打印去重信息
    duplicate_count = len(all_commits) - len(unique_commits)
    print(f"去重完成，共删除了 {duplicate_count} 个重复 commit，剩余 {len(unique_commits)} 个唯一 commit。")
    print(f"去重条件：hash + message + modified_files 三个字段都必须完全相同")
    print(f"去重策略：优先保留在代码库列表中顺序靠前的代码库中的commit")
    
    # 打印各代码库的重复commit被删除情况
    if duplicate_stats["by_repo"]:
        print("\n各代码库被删除的重复commit统计:")
        for repo_name in ordered_repo_names:
            if repo_name in duplicate_stats["by_repo"]:
                removed_count = duplicate_stats["by_repo"][repo_name]
                print(f"  {repo_name}: {removed_count} 个commit被删除")
    
    # 保存去重后的文件
    dedup_file_path = os.path.join(root_path, "all_has_file_deduplicated.json")
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
    for repo_name, repo_commits in tqdm(commits_by_repo.items(), desc="分发结果"):
        output_path = os.path.join(knowledge_base_path, repo_name, "has_file_deduplicate.json")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存到对应的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(repo_commits, f, indent=4, ensure_ascii=False)
        
        # 计算去重率
        original_count = repository_commit_counts.get(repo_name, 0)
        dedup_count = len(repo_commits)
        removed_count = duplicate_stats["by_repo"].get(repo_name, 0)
        
        if original_count > 0:
            dedup_rate = removed_count / original_count * 100
        else:
            dedup_rate = 0
            
        print(f"代码库 {repo_name}: 原始 {original_count} 个，去重后 {dedup_count} 个，删除重复 {removed_count} 个，去重率 {dedup_rate:.2f}%")
    
    print("\n全部处理完成！")
    
    return {
        "total_original": len(all_commits),
        "total_deduplicated": len(unique_commits),
        "deduplication_rate": (len(all_commits) - len(unique_commits)) / len(all_commits) * 100 if all_commits else 0,
        "duplicate_stats": duplicate_stats,
        "processed_repos": processed_repos
    }




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
    root_result_path = os.path.join(config.root_path, "knowledge_base_all")
    
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



def filter_optimization_commits_2():
    """
    遍历 root_path/knowledge_base 中的所有子文件夹，处理每个子文件夹中的 is_opt_llm_2.json 文件。
    从输入文件中筛选出 is_opt_ds_simple 和 is_general_ds_simple 均为 true 的 commit，并写入到输出文件 is_opt_final_2.json。
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
        input_path = os.path.join(repo_path, "is_opt_llm_2.json")
        
        # 输出文件路径
        output_path = os.path.join(repo_path, "is_opt_final_2.json")
        
        # 如果输入文件存在，则处理
        if os.path.exists(input_path):
            try:
                # 读取输入文件
                with open(input_path, "r") as f:
                    commit_data = json.load(f)
                
                # 筛选出 is_opt_ds_simple 和 is_general_ds_simple 均为 true 的 commit
                filtered_commits = [
                    commit for commit in commit_data
                    if (commit.get("is_opt_ds_simple", "").lower() == "true" and 
                        commit.get("is_general_ds_simple", "").lower() == "true")
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
            return repo_name, 0, "未找到 is_opt_llm_2.json 文件"
    
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




def filter_commits_by_criteria():
    """
    处理 config.root_path/knowledge_base 中各仓库的 one_func.json 文件。
    实现两个筛选条件：
    1. 按修改总行数筛选，只保留在指定范围内的commit
    2. 排除只修改了测试函数的commit
    
    筛选结果覆盖原始 one_func.json 文件
    """
    # 筛选配置
    filter_by_line_count = True  # 是否按修改行数筛选
    min_lines = 1  # 修改行数下限
    max_lines = 50  # 修改行数上限
    
    exclude_test_funcs = True  # 是否排除测试函数
    # 测试函数名列表，可按需扩展
    test_function_patterns = [
        "TEST_F", "TEST", "test_", "Test", "_test", 
        "FIXTURE", "fixture", "mock", "Mock", 
        "stub", "Stub", "fake", "Fake", "benchmark"
    ]
    
    max_workers = 128
    
    # 获取知识库根路径
    root_result_path = os.path.join(config.root_path, "knowledge_base")
    
    # 获取仓库列表
    repositories = [repo_name for repo_name in os.listdir(root_result_path)
                   if os.path.isdir(os.path.join(root_result_path, repo_name))]
    
    print(f"找到 {len(repositories)} 个代码库，开始筛选提交...")
    print(f"使用 {max_workers} 个并行线程进行处理")
    
    if filter_by_line_count:
        print(f"启用修改行数筛选: {min_lines} - {max_lines} 行")
    
    if exclude_test_funcs:
        print(f"启用测试函数排除: {', '.join(test_function_patterns)}")
    
    # 检查函数名是否是测试函数
    def is_test_function(func_name):
        if not exclude_test_funcs:
            return False
        
        if not func_name:
            return False
            
        for pattern in test_function_patterns:
            if pattern.lower() in func_name.lower() or func_name.startswith(pattern):
                return True
        return False
    
    # 处理单个仓库的内部函数
    def process_repo(repo_name):
        repo_path = os.path.join(root_result_path, repo_name)
        
        # 文件路径 (输入输出同一文件)
        json_path = os.path.join(repo_path, "one_func.json")
        
        # 如果文件存在，则处理
        if os.path.exists(json_path):
            try:
                # 读取文件
                with open(json_path, "r", encoding="utf-8") as f:
                    commit_data = json.load(f)
                
                # 记录原始提交数量
                original_count = len(commit_data)
                
                # 筛选出符合条件的commit
                filtered_commits = []
                for commit in commit_data:
                    # 筛选条件1: 检查修改行数
                    if filter_by_line_count:
                        total_changed_lines = commit.get("total_changed_lines", 0)
                        if not (min_lines <= total_changed_lines <= max_lines):
                            continue
                    
                    # 筛选条件2: 排除测试函数
                    if exclude_test_funcs and "modified_func_count" in commit and commit.get("modified_func_count") == 1:
                        # 只修改了一个函数的情况下，检查是否是测试函数
                        modified_funcs = commit.get("modified_func", [])
                        if modified_funcs and is_test_function(modified_funcs[0]):
                            continue
                    
                    # 通过所有筛选条件，加入结果列表
                    filtered_commits.append(commit)
                
                # 将筛选结果覆盖原文件
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_commits, f, indent=4, ensure_ascii=False)
                
                return repo_name, len(filtered_commits), original_count, True
            except Exception as e:
                return repo_name, 0, 0, f"错误: {e}"
        else:
            return repo_name, 0, 0, "未找到 one_func.json 文件"
    
    # 初始化统计信息
    total_success = 0
    total_filtered_commits = 0
    total_original_commits = 0
    
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
                    repo_name, filtered_count, original_count, result = future.result()
                    
                    # 更新进度条
                    pbar.update(1)
                    
                    # 处理结果
                    if result is True:
                        total_success += 1
                        total_filtered_commits += filtered_count
                        total_original_commits += original_count
                        if filtered_count > 0:
                            pbar.set_postfix_str(f"当前: {repo_name} ({filtered_count}/{original_count}个提交)")
                    else:
                        tqdm.write(f"[{repo_name}] {result}")
                except Exception as e:
                    pbar.update(1)
                    tqdm.write(f"[{repo_name}] 处理异常: {str(e)}")
    
    # 计算筛选率
    filter_rate = (total_original_commits - total_filtered_commits) / total_original_commits * 100 if total_original_commits > 0 else 0
    
    print(f"\n筛选完成! 成功处理 {total_success} 个代码库")
    print(f"原始提交数: {total_original_commits}，筛选后提交数: {total_filtered_commits}，筛选掉 {filter_rate:.2f}% 的提交")




def aggregate_final_commits():
    """
    遍历 root_path/knowledge_base 中的所有子文件夹，读取每个子文件夹中的 is_opt_final.json 文件。
    将所有符合条件的 commit 汇总到 root_path/all_is_opt_final.json 文件中，并删除每个 commit 中的 all_functions 字段。
    同时每个 commit 添加字段 repository_name，记录该 commit 所在的仓库名称。
    """
    # 定义结果路径
    root_result_path = os.path.join(config.root_path, "knowledge_base_all")
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



def tmp_delete_file():
    """
    遍历 config.root_path/knowledge_base 中的所有代码库，
    删除每个代码库中的 has_file_with_func.json 文件。
    """
    knowledge_base_path = os.path.join(config.root_path, "knowledge_base")
    
    # 检查 knowledge_base 目录是否存在
    if not os.path.exists(knowledge_base_path):
        print(f"目录 {knowledge_base_path} 不存在")
        return
    
    deleted_count = 0
    skipped_count = 0
    
    for repo_name in os.listdir(knowledge_base_path):
        repo_path = os.path.join(knowledge_base_path, repo_name)
        # 仅处理目录（代码库）
        if os.path.isdir(repo_path):
            target_file = os.path.join(repo_path, "has_file_with_func.json")
            
            # 检查目标文件是否存在
            if os.path.exists(target_file):
                try:
                    os.remove(target_file)
                    print(f"成功删除文件: {target_file}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除 {target_file} 时出错: {e}")
            else:
                print(f"文件 {target_file} 不存在，跳过仓库：{repo_name}")
                skipped_count += 1
    
    print(f"\n删除操作完成:")
    print(f"  成功删除: {deleted_count} 个文件")
    print(f"  跳过: {skipped_count} 个仓库（文件不存在）")



# get_all_commits()

# filter_one_file_commits()
# filter_c_language_commits()

# aggregate_and_deduplicate_has_file()
# aggregate_and_deduplicate_one_func()

# get_has_file()
# copy_has_file()

# filter_one_func_commits()
# copy_one_func_file()

# copy_func_name_file(merge_mode=False)  # 直接覆盖模式
# copy_func_name_file(merge_mode=True)   # 智能合并模式

# filter_optimization_commits()
# filter_optimization_commits_2()

aggregate_final_commits()

# tmp_delete_file()