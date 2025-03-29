import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 最大并行数
MAX_WORKERS = 8

# 是否重新生成已存在的解释文件
# False：使用现有结果
# True：全部重新生成
REGENERATE_EXISTING = False


def call_llm(client, prompt):
    """
    通用的 llm 调用函数，返回回复内容（去除可能的 Markdown 包裹）
    """
    response = client.chat.completions.create(
        model=config.xmcp_o3_mini_model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content.strip()
    lines = content.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip().startswith("```"):
        content = "\n".join(lines[1:-1]).strip()
    return content

def create_commit_explanation(client, commit_hash, commit_message, commit_dir):
    """
    为提交创建解释：结合commit message和函数修改
    """
    exp_path = os.path.join(commit_dir, 'exp.txt')
    
    # 检查是否需要重新生成
    if os.path.exists(exp_path) and not REGENERATE_EXISTING:
        print(f"提交 {commit_hash} 的解释文件已存在，跳过")
        return

    before_func_path = os.path.join(commit_dir, 'before_func.txt')
    after_func_path = os.path.join(commit_dir, 'after_func.txt')
    
    if not os.path.exists(before_func_path) or not os.path.exists(after_func_path):
        print(f"警告: 提交 {commit_hash} 缺少函数修改前后文件，仅使用commit message生成解释")
        with_func_compare = False
    else:
        with_func_compare = True
        with open(before_func_path, 'r', encoding='utf-8') as f:
            before_func = f.read()
        with open(after_func_path, 'r', encoding='utf-8') as f:
            after_func = f.read()
    
    # 构建提示
    if with_func_compare:
        prompt = (
            "请你结合以下提交信息和函数修改前后的代码，用中文详细总结这次代码提交的主要修改思路、目的和预期的优化效果：\n\n"
            f"提交信息(Commit Message):\n{commit_message}\n\n"
            f"函数修改前:\n{before_func}\n\n"
            f"函数修改后:\n{after_func}\n\n"
            "请详细分析并总结:"
        )
    else:
        prompt = (
            "请你解析以下提交信息，用中文详细总结这次代码提交的主要修改思路、目的和预期的优化效果：\n\n"
            f"提交信息(Commit Message):\n{commit_message}\n\n"
            "请详细分析并总结:"
        )
    
    try:
        explanation = call_llm(client, prompt)
        # 保存解释结果
        with open(exp_path, 'w', encoding='utf-8') as f:
            f.write(explanation)
        return True
    except Exception as e:
        print(f"请求 LLM 生成提交解释时出错（Commit: {commit_hash}）：{e}")
        return False

def process_commit_dir(client, commit_dir, commit_info_dict):
    """
    处理单个提交目录：为提交生成解释
    """
    commit_hash = os.path.basename(commit_dir)
    
    # 获取提交信息
    commit_info = commit_info_dict.get(commit_hash)
    if not commit_info:
        print(f"警告：未找到提交 {commit_hash} 的信息")
        return False
    
    commit_message = commit_info.get('message', '')
    if not commit_message:
        print(f"警告：提交 {commit_hash} 缺少提交信息")
        return False
    
    # 创建提交解释
    result = create_commit_explanation(client, commit_hash, commit_message, commit_dir)
    return result

def main():
    """
    主函数：遍历所有提交目录并生成提交解释
    """
    repo_name = "rocksdb"  # 可按需修改或参数化库名
    
    modified_file_dir = os.path.join(
        config.root_path,
        'benchmark',
        repo_name,
        'modified_file'
    )
    
    if not os.path.exists(modified_file_dir):
        print(f"错误：目录 {modified_file_dir} 不存在。")
        sys.exit(1)
    
    # 读取test_result.json以获取提交信息
    test_json_path = os.path.join(
        config.root_path,
        'benchmark',
        repo_name,
        'test_result.json'
    )
    
    commit_info_dict = {}
    if os.path.exists(test_json_path):
        try:
            with open(test_json_path, 'r', encoding='utf-8') as f:
                commits = json.load(f)
                for commit in commits:
                    commit_hash = commit.get('hash')
                    if commit_hash:
                        commit_info_dict[commit_hash] = commit
            print(f"成功从 {test_json_path} 读取 {len(commit_info_dict)} 条提交信息")
        except Exception as e:
            print(f"警告：读取 {test_json_path} 时出错：{e}")
            print("继续处理，但将无法获取完整的提交信息")
    else:
        print(f"错误：文件 {test_json_path} 不存在")
        sys.exit(1)
    
    # 获取所有提交目录
    commit_dirs = []
    for item in os.listdir(modified_file_dir):
        item_path = os.path.join(modified_file_dir, item)
        if os.path.isdir(item_path):
            commit_dirs.append(item_path)
    
    if not commit_dirs:
        print(f"警告：在 {modified_file_dir} 中未找到任何提交目录")
        sys.exit(0)
    
    print(f"找到 {len(commit_dirs)} 个提交目录待处理")
    
    # 初始化OpenAI客户端
    client = OpenAI(
        base_url=config.xmcp_base_url,
        api_key=config.xmcp_api_key,
    )
    
    # 使用ThreadPool并行处理所有提交目录
    results = {"success": 0, "failed": 0, "skipped": 0}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_commit = {
            executor.submit(process_commit_dir, client, commit_dir, commit_info_dict): commit_dir 
            for commit_dir in commit_dirs
        }
        
        # 显示进度条
        for future in tqdm(as_completed(future_to_commit), total=len(future_to_commit), 
                          desc="解释提交信息", unit="commit"):
            commit_dir = future_to_commit[future]
            commit_hash = os.path.basename(commit_dir)
            try:
                result = future.result()
                if result is True:
                    results["success"] += 1
                elif result is False:
                    results["failed"] += 1
                else:
                    results["skipped"] += 1
            except Exception as e:
                print(f"处理提交 {commit_hash} 时出错: {e}")
                results["failed"] += 1
    
    # 输出统计信息
    print("\n提交解释生成完成!")
    print(f"总计: {len(commit_dirs)} 个提交")
    print(f"成功: {results['success']}")
    print(f"失败: {results['failed']}")
    print(f"跳过: {results['skipped']}")

if __name__ == "__main__":
    main()