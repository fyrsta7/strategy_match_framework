import subprocess
import sys
import os
from typing import List, Optional


def run_command(command: List[str], cwd: Optional[str] = None) -> bool:
    """
    运行命令并返回是否成功。
    """
    try:
        print(f"运行命令: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"命令失败：{' '.join(command)}")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"运行命令时出错: {' '.join(command)}")
        print(e)
        return False


def get_commits(repo_path: str, old_commit: str, new_commit: str) -> List[str]:
    """
    获取两个提交之间的所有提交，按从旧到新的顺序排列。
    """
    cmd = ['git', 'rev-list', '--reverse', f'{old_commit}..{new_commit}']
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        commits = result.stdout.strip().split('\n')
        return commits
    except subprocess.CalledProcessError as e:
        print("获取提交列表失败:")
        print(e.stderr)
        sys.exit(1)


def reset_repo(repo_path: str) -> bool:
    """
    放弃当前所有修改，重置仓库到HEAD并清理未跟踪的文件。
    """
    cmd_reset = ['git', 'reset', '--hard']
    cmd_clean = ['git', 'clean', '-fdx']
    if not run_command(cmd_reset, cwd=repo_path):
        print("git reset --hard 失败。")
        return False
    if not run_command(cmd_clean, cwd=repo_path):
        print("git clean -fdx 失败。")
        return False
    return True


def checkout_commit(repo_path: str, commit: str) -> bool:
    """
    放弃当前所有修改并检出到指定的提交。
    """
    # 放弃所有修改
    if not reset_repo(repo_path):
        print("重置仓库失败，无法检出提交。")
        return False

    # 检出指定的提交
    cmd = ['git', 'checkout', commit]
    return run_command(cmd, cwd=repo_path)


def modify_files(repo_path: str, files: List[str], include_statement: str = '#include <cstdint>') -> bool:
    """
    在指定文件的 #pragma once 后面添加 #include <cstdint>。
    """
    for file_rel_path in files:
        file_path = os.path.join(repo_path, file_rel_path)
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 检查是否已经包含了 #include <cstdint>
            if any(include_statement in line for line in lines):
                print(f"文件 {file_rel_path} 已包含 '{include_statement}'。")
                continue

            # 查找 #pragma once 的位置
            pragma_once_index = -1
            for i, line in enumerate(lines):
                if line.strip() == '#pragma once':
                    pragma_once_index = i
                    break

            if pragma_once_index == -1:
                print(f"在文件 {file_rel_path} 中未找到 '#pragma once'。")
                return False

            # 在 #pragma once 后插入 #include <cstdint>
            lines.insert(pragma_once_index + 1, include_statement + '\n')

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f"已在文件 {file_rel_path} 中插入 '{include_statement}'。")
        except Exception as e:
            print(f"修改文件 {file_rel_path} 时出错: {e}")
            return False
    return True


def build_and_test(repo_path: str) -> bool:
    """
    执行编译和测试命令。
    """
    commands = [
        ['make', 'clean'],
        ['make', 'static_lib', f'-j{os.cpu_count()}', 'DEBUG_LEVEL=0'],
        ['make', 'db_bench', f'-j{os.cpu_count()}', 'DEBUG_LEVEL=0'],
        ['rm', '-rf', '/tmp/rocksdb_test'],
        ['./db_bench', '--benchmarks=fillseq,readrandom', '--db=/tmp/rocksdb_test', '--num=1000000']
    ]

    for cmd in commands:
        success = run_command(cmd, cwd=repo_path)
        if not success:
            return False
    return True


def binary_search_commits(repo_path: str, commits: List[str], required_files: List[str]) -> Optional[str]:
    """
    在提交列表中进行二分查找，找到第一个可以成功运行的提交。
    """
    left = 0
    right = len(commits) - 1
    first_success = None

    while left <= right:
        mid = (left + right) // 2
        commit = commits[mid]
        print(f"\n检查提交 {commit} ({mid + 1}/{len(commits)})")

        # 检出提交
        if not checkout_commit(repo_path, commit):
            print(f"无法检出到提交 {commit}, 假设失败")
            # 搜索右半部分
            left = mid + 1
            continue

        # 修改指定文件
        if not modify_files(repo_path, required_files):
            print(f"修改文件失败，假设提交 {commit} 运行失败，搜索右半部分")
            left = mid + 1
            continue

        # 执行编译和测试
        if not build_and_test(repo_path):
            print(f"提交 {commit} 运行失败，搜索右半部分")
            left = mid + 1
        else:
            print(f"提交 {commit} 运行成功，记录并搜索左半部分")
            first_success = commit
            right = mid - 1

    return first_success


def main():
    repo_path = "/home/zyw/llm_on_code/llm_on_code_optimization/repository/rocksdb"
    old_commit = "f9cfc6a808c9dc3ab7366edb10368559155d5172"
    new_commit = "687a2a0d9ad5b0a3588e331ecd15317f3384def0"

    # 检查工作区是否干净
    status_cmd = ['git', 'status', '--porcelain']
    try:
        result = subprocess.run(
            status_cmd,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        if result.stdout.strip():
            print("工作区不干净，请确保工作区干净后再运行脚本。")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("无法获取 git 状态:")
        print(e.stderr)
        sys.exit(1)

    # 获取提交列表
    commits = get_commits(repo_path, old_commit, new_commit)
    if not commits:
        print("没有找到两个提交之间的任何提交。")
        sys.exit(1)

    print(f"总共有 {len(commits)} 个提交需要检查。")

    # 定义需要修改的文件列表
    required_files = [
        "table/block_based/data_block_hash_index.h",
        "util/string_util.h",
    ]

    # 执行二分查找
    first_success = binary_search_commits(repo_path, commits, required_files)

    if first_success:
        print(f"\n第一个可以成功运行的提交是: {first_success}")
    else:
        print("\n在给定的提交范围内没有找到可以成功运行的提交。")

    # 可选：返回到new_commit以保持代码库状态
    if first_success:
        if not checkout_commit(repo_path, new_commit):
            print("无法检出到最终提交。")
            sys.exit(1)
        print(f"已返回到最终提交: {new_commit}")
    else:
        print("未检测到成功的提交，因此不返回到最终提交。")


if __name__ == "__main__":
    main()