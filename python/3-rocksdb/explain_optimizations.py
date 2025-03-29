import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 全局配置
MODELS = ['deepseek-v3', 'gpt-4o']  # 需要处理的模型列表
MAX_WORKERS = 64  # 最大并行数

# 是否重新生成已存在的解释文件
# False：使用现有结果
# True：全部重新生成
REGENERATE_EXISTING = False

# 在 rocksdb/modified_file/<commit hash>/FILE_DIR/<model name>/ 中存储 llm 的优化结果
FILE_DIR = "bm25"

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

def process_optimized_file(client, commit_dir, model_name, file_name):
    """
    处理单个优化文件：生成解释并保存
    """
    base_path = os.path.join(commit_dir, FILE_DIR, model_name)
    input_path = os.path.join(base_path, file_name)
    exp_file_name = f"exp_{file_name}"
    output_path = os.path.join(base_path, exp_file_name)
    
    # 检查是否需要重新生成
    if os.path.exists(output_path) and not REGENERATE_EXISTING:
        return {"status": "skipped", "path": output_path}
    
    # 读取原始函数和优化后的函数
    before_func_path = os.path.join(commit_dir, 'before_func.txt')
    
    if not os.path.exists(before_func_path):
        return {"status": "error", "message": f"原始函数文件不存在: {before_func_path}"}
    
    if not os.path.exists(input_path):
        return {"status": "error", "message": f"优化后函数文件不存在: {input_path}"}
    
    try:
        with open(before_func_path, 'r', encoding='utf-8') as f:
            original_function = f.read()
        with open(input_path, 'r', encoding='utf-8') as f:
            optimized_function = f.read()
        
        func_prompt = (
            "请根据下面提供的函数修改前后的内容，用一句中文总结说明：\n"
            "1. 函数主要做了哪些修改？\n"
            "2. 这些修改的目的和优化点是什么？\n\n"
            "【修改前的函数】:\n"
            f"{original_function}\n\n"
            "【修改后的函数】:\n"
            f"{optimized_function}\n\n"
            "一句总结："
        )
        
        func_summary = call_llm(client, func_prompt)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存解释结果
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(func_summary)
        
        return {"status": "success", "path": output_path}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

def find_optimization_files(commit_dirs):
    """
    查找所有需要处理的优化文件
    返回: [(commit_dir, model_name, file_name), ...]
    """
    optimization_files = []
    
    for commit_dir in commit_dirs:
        commit_hash = os.path.basename(commit_dir)
        
        for model_name in MODELS:
            model_dir = os.path.join(commit_dir, FILE_DIR, model_name)
            if not os.path.exists(model_dir):
                continue
            
            for file_name in os.listdir(model_dir):
                # 只处理数字命名的文件(例如 1.txt, 2.txt)
                if file_name.endswith('.txt') and file_name[:-4].isdigit():
                    # 跳过已经是解释文件的文件
                    if file_name.startswith('exp_'):
                        continue
                    optimization_files.append((commit_dir, model_name, file_name))
                elif file_name.isdigit():
                    # 处理不带后缀的数字文件
                    file_name_with_ext = f"{file_name}.txt"
                    optimization_files.append((commit_dir, model_name, file_name_with_ext))
    
    return optimization_files

def main():
    """
    主函数：遍历所有提交目录，查找并处理所有优化文件
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
    
    # 获取所有提交目录
    commit_dirs = []
    for item in os.listdir(modified_file_dir):
        item_path = os.path.join(modified_file_dir, item)
        if os.path.isdir(item_path):
            commit_dirs.append(item_path)
    
    if not commit_dirs:
        print(f"警告：在 {modified_file_dir} 中未找到任何提交目录")
        sys.exit(0)
    
    # 查找所有需要处理的优化文件
    optimization_files = find_optimization_files(commit_dirs)
    print(f"找到 {len(optimization_files)} 个优化文件需要解释")
    
    if not optimization_files:
        print("没有找到需要处理的优化文件")
        sys.exit(0)
    
    # 初始化OpenAI客户端
    client = OpenAI(
        base_url=config.xmcp_base_url,
        api_key=config.xmcp_api_key,
    )
    
    # 结果统计
    results = {"success": 0, "error": 0, "skipped": 0}
    
    # 并行处理所有优化文件
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_optimized_file, client, commit_dir, model_name, file_name): 
            (commit_dir, model_name, file_name)
            for commit_dir, model_name, file_name in optimization_files
        }
        
        # 显示进度条
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), 
                           desc="解释优化结果", unit="file"):
            commit_dir, model_name, file_name = future_to_file[future]
            try:
                result = future.result()
                status = result.get("status")
                
                if status == "success":
                    results["success"] += 1
                elif status == "skipped":
                    results["skipped"] += 1
                else:
                    results["error"] += 1
                    commit_hash = os.path.basename(commit_dir)
                    print(f"错误: {commit_hash}/{model_name}/{file_name}: {result.get('message', '未知错误')}")
            except Exception as e:
                results["error"] += 1
                commit_hash = os.path.basename(commit_dir)
                print(f"处理 {commit_hash}/{model_name}/{file_name} 时出错: {e}")
    
    # 输出统计信息
    print("\n优化解释生成完成!")
    print(f"总计: {len(optimization_files)} 个优化文件")
    print(f"成功: {results['success']}")
    print(f"失败: {results['error']}")
    print(f"跳过: {results['skipped']}")

if __name__ == "__main__":
    main()