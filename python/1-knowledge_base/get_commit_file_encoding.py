import os
import subprocess
import chardet
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def detect_file_encoding(file_content):
    """
    检测文件内容的编码格式。
    """
    result = chardet.detect(file_content)
    return result['encoding']

def get_file_content(repo_path, commit_hash, file_path):
    """
    获取指定 commit 中某个文件的内容。
    """
    try:
        # 获取文件内容
        show_command = f"git -C {repo_path} show {commit_hash}:{file_path}"
        content = subprocess.check_output(show_command, shell=True, stderr=subprocess.PIPE)
        return content
    except subprocess.CalledProcessError as e:
        print(f"无法获取文件 {file_path} 的内容: {e.stderr.decode('utf-8', errors='ignore')}")
        return None

def analyze_commit_encodings(repo_name, commit_hash):
    """
    分析指定代码库和 commit 中修改文件的编码格式。
    """
    # 根路径
    root_path = config.root_path
    repo_path = os.path.join(root_path, "repository", repo_name)
    result_path = os.path.join(root_path, "result", repo_name)
    json_file_path = os.path.join(result_path, "c_language.json")

    # 检查代码库是否存在
    if not os.path.isdir(repo_path):
        print(f"代码库 {repo_name} 不存在。")
        return

    # 检查 c_language.json 是否存在
    if not os.path.exists(json_file_path):
        print(f"c_language.json 文件不存在于 {result_path}。")
        return

    # 读取 c_language.json 文件
    with open(json_file_path, 'r') as f:
        commits = json.load(f)

    # 查找指定 commit 的信息
    target_commit = None
    for commit in commits:
        if commit['hash'] == commit_hash:
            target_commit = commit
            break

    if not target_commit:
        print(f"未找到 commit {commit_hash} 的信息。")
        return

    # 获取修改的文件列表
    modified_files = target_commit.get('modified_files', [])
    if not modified_files:
        print(f"commit {commit_hash} 中没有修改的文件。")
        return

    print(f"正在分析代码库 {repo_name} 的 commit {commit_hash} 中修改文件的编码...")

    # 遍历每个修改的文件并检测编码
    for file_path in modified_files:
        # 获取文件内容
        file_content = get_file_content(repo_path, commit_hash, file_path)
        if file_content is None:
            continue

        # 检测文件编码
        encoding = detect_file_encoding(file_content)
        print(f"文件: {file_path} | 编码: {encoding}")

if __name__ == "__main__":
    # 输入代码库名称和 commit hash
    repo_name = "opencv"
    commit_hash = "0bc9a0db18e750f3c66f72948cb714a0c9d540be"

    # 分析 commit 中修改文件的编码
    analyze_commit_encodings(repo_name, commit_hash)