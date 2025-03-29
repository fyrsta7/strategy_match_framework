import os
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 定义路径
repository_dir = os.path.join(config.root_path, "repository")  # repository 文件夹的路径
result_dir = os.path.join(config.root_path, "result")  # result 文件夹的路径

# 支持的 README 文件名变体
readme_names = [
    "README.md",  # 优先使用根目录的 README.md
    "readme.md",
    "README.MD",
    "ReadMe.md",
    "Readme.md",
    "README.markdown",
    "readme.markdown",
    "README.rst",
    "readme.rst",
    "ReadMe.rst",
    "README",
    "readme",
    "README.txt",
    "readme.txt",
    "README.adoc",
    "README.org",
    "README.asciidoc",
    "README.Md"
]

def find_and_copy_readme(repo_path, repo_name):
    """
    在代码库文件夹中查找 README 文件，并复制到 result 文件夹中。
    """
    # 优先查找根目录的 README 文件
    for readme_name in readme_names:
        readme_path = os.path.join(repo_path, readme_name)
        if os.path.exists(readme_path):
            # 创建目标文件夹
            target_dir = os.path.join(result_dir, repo_name)
            os.makedirs(target_dir, exist_ok=True)

            # 复制并重命名 README 文件
            target_path = os.path.join(target_dir, "README.md")
            shutil.copy2(readme_path, target_path)
            print(f"Found and copied README for '{repo_name}' to '{target_path}'.")
            return True

    # 如果根目录没有 README 文件，检查 docs/README.md
    docs_readme_path = os.path.join(repo_path, "docs", "README.md")
    if os.path.exists(docs_readme_path):
        # 创建目标文件夹
        target_dir = os.path.join(result_dir, repo_name)
        os.makedirs(target_dir, exist_ok=True)

        # 复制并重命名 README 文件
        target_path = os.path.join(target_dir, "README.md")
        shutil.copy2(docs_readme_path, target_path)
        print(f"Found and copied docs/README.md for '{repo_name}' to '{target_path}'.")
        return True

    # 如果没有找到 README 文件
    print(f"No README file found for '{repo_name}'.")
    return False

def main():
    # 确保 result 文件夹存在
    os.makedirs(result_dir, exist_ok=True)

    # 遍历 repository 文件夹中的所有代码库
    for repo_name in os.listdir(repository_dir):
        repo_path = os.path.join(repository_dir, repo_name)

        # 确保是文件夹
        if os.path.isdir(repo_path):
            find_and_copy_readme(repo_path, repo_name)

    print("All repositories processed.")

if __name__ == "__main__":
    main()