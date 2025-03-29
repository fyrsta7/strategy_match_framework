import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 全局变量：设置是否在 after_func_{CURRENT_MODEL}.txt 已存在时重新生成优化结果
# 设置为 True 时，即使文件存在，也会重新调用 LLM 生成优化结果
# 设置为 False 时，如果文件已存在，则跳过该提交的优化过程
REGENERATE_EXISTING = False

# prompt中参考示例的数量
MAX_EXAMPLES = 4

# 定义需要处理的模型（示例）
MODELS = ['gpt-4o']
# 支持的模型列表
SUPPORTED_MODELS = ['deepseek-v3', 'deepseek-reasoner', 'gpt-4o', 'o1-mini', 'o3-mini']

# 放 commit 相关文件的路径
COMMIT_DIR = config.root_path + "benchmark/{repo}/modified_file/{commit}"
# 放结果文件的子文件夹名
OUTPUT_DIRNAME = "langchain+cosine"
# 记录llm优化结果的文件名
OUTPUT_FILENAME = f"related_{MAX_EXAMPLES}_diff" + "_{model}.txt"

# 指定待优化commit列表文件（内容为commit对象数组，包含 "hash" 等信息）
COMMIT_LIST_FILE = os.path.join(config.root_path, 'benchmark', 'rocksdb', 'filtered_test_result.json')

# 加载全局的相似性数据
similarity_path = os.path.join(config.root_path, "commit_similarity_langchain+cosine.json")

def get_client_and_model_info(model):
    """
    根据传入的模型名称返回 (client, model_name)。
    注意：不同模型需要不同的调用参数和变量。
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Error: Unsupported model '{model}'. Supported: {SUPPORTED_MODELS}")

    if model == 'deepseek-v3':
        base_url = config.deepseek_base_url
        api_key = config.deepseek_api_key
        # deepseek-v3 使用 'deepseek-chat' 作为 model 名称
        model_name = 'deepseek-chat'
    elif model == 'deepseek-reasoner':
        base_url = config.deepseek_base_url
        api_key = config.deepseek_api_key
        model_name = 'deepseek-reasoner'
    else:
        # 对于 'gpt-4o'、'o1-mini' 或 'o3-mini'
        base_url = config.closeai_base_url
        api_key = config.closeai_api_key
        model_name = model

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    except Exception as e:
        raise RuntimeError(f"Client initialization failed for model {model}: {e}")
    
    return client, model_name

def load_example_data(commit_info):
    """
    加载指定commit的diff.txt文件。
    文件路径：root_path/result/<repository_name>/modified_file/<commit_hash>/
    返回字典包含{'diff': 内容}；若加载失败返回None。
    """
    result_dir = os.path.join(
        config.root_path,
        'result',
        commit_info["repository_name"],
        'modified_file',
        commit_info["commit_hash"]
    )
    
    data = {}
    try:
        with open(os.path.join(result_dir, 'diff.txt'), 'r', encoding='utf-8') as f:
            data['diff'] = f.read().strip()
        return data
    except Exception as e:
        print(f"Failed to load example {commit_info['commit_hash']}: {str(e)}")
        return None

def build_examples_section(commit_hash, repo_name, similarity_data):
    """
    构造参考示例部分文本，包含前MAX_EXAMPLES个相似commit的diff.txt内容。
    处理流程：
      1. 在相似性数据中找到与query_commit匹配的记录；
      2. 从该记录中选择相似度最高的MAX_EXAMPLES个commit；
      3. 再将所选commit按照相似度从低到高排序；
      4. 对每个示例，只保留diff信息，按固定格式拼接后返回。
    """
    query_entry = next(
        (item for item in similarity_data 
         if item["query_commit"]["commit_hash"] == commit_hash
         and item["query_commit"]["repository_name"] == repo_name),
        None
    )
    
    if not query_entry:
        return ""
    
    valid_examples = query_entry["similar_commits"]
    
    # 先按相似度从高到低选择top n，然后反转为升序排序
    sorted_examples = sorted(
        valid_examples,
        key=lambda x: x["similarity_score"],
        reverse=True
    )[:MAX_EXAMPLES][::-1]
    
    examples = []
    for ex in sorted_examples:
        if (ex_data := load_example_data(ex)):
            examples.append({
                "similarity": ex["similarity_score"],
                # 不包含before信息，只保留diff
                "diff": ex_data['diff']
            })
    
    example_text = ""
    for idx, ex in enumerate(examples):
        example_text += (
            f"==== Example {idx+1} (Similarity: {ex['similarity']:.1f}) ====\n"
            "Optimization Changes:\n"
            f"```diff\n{ex['diff']}\n```\n"
            "------------------------\n\n"
        )
    
    return example_text.strip()

def generate_prompt(target_function, examples_section):
    """
    构造prompt，要求LLM根据参考示例（仅包含diff信息）优化目标函数。
    prompt内容如下：
      - 优化要求说明：要求保持相同功能，返回完整优化后的函数，不附加额外文字。
      - 参考示例部分（按相似度升序排列diff信息）。
      - 待优化的目标函数代码。
    """
    return f"""Optimize the following C/C++ function based on the reference examples. Requirements:
1. Maintain EXACT functionality.
2. Return ONLY the complete optimized function code.
3. No explanations or comments allowed.

Reference optimization examples (ordered by ascending similarity):
{examples_section if examples_section else "No relevant examples"}

Function to optimize:
```cpp
{target_function}
```

Return ONLY the optimized function code without any additional text."""

def optimize_function(client, repo_name, commit_hash, similarity_data, current_model, model_name):
    """
    针对每个待优化的commit进行处理：
      1. 从benchmark目录中读取目标函数（before_func.txt）。
      2. 构建包含参考示例的prompt。
      3. 调用LLM优化该函数，并将结果保存在输出文件 related_{MAX_EXAMPLES}_{current_model}.txt中。
    """
    # 对于待优化的 commit，其目录位于 benchmark 下
    commit_dir = COMMIT_DIR.format(repo = repo_name, commit = commit_hash)
    
    # 创建输出目录
    output_dir = commit_dir + "/" + OUTPUT_DIRNAME
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径，文件名包含当前模型标识
    output_path = os.path.join(output_dir, OUTPUT_FILENAME.format(model = current_model))
    
    if os.path.exists(output_path) and not REGENERATE_EXISTING:
        print(f"Skipping existing commit: {commit_hash} (model: {current_model})")
        return
    
    before_func_path = os.path.join(commit_dir, 'before_func.txt')
    if not os.path.exists(before_func_path):
        print(f"Missing source file: {before_func_path}")
        return
    
    try:
        with open(before_func_path, 'r', encoding='utf-8') as f:
            target_function = f.read().strip()
    except Exception as e:
        print(f"Failed to read target function for commit {commit_hash}: {str(e)}")
        return
    
    examples_section = build_examples_section(commit_hash, repo_name, similarity_data)
    full_prompt = generate_prompt(target_function, examples_section)
    
    try:
        if current_model in ["o1-mini", "o3-mini"]:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
                max_tokens=8192
            )
        optimized_code = response.choices[0].message.content
        # 清理返回代码中的 Markdown 标识
        optimized_code = optimized_code.replace('```cpp', '').replace('```c', '').replace('```', '').strip()
        # 保存优化结果
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(optimized_code)
        status = "Regenerated" if os.path.exists(output_path) else "Created"
        print(f"{status} optimization for commit {commit_hash} (model: {current_model}) at {output_path}")
        
    except Exception as e:
        print(f"API request failed for commit {commit_hash} (model: {current_model}): {str(e)}")

def main():
    repo_name = "rocksdb"
    
    try:
        with open(similarity_path, 'r', encoding='utf-8') as f:
            similarity_data = json.load(f)
    except Exception as e:
        print(f"Failed to load similarity data: {str(e)}")
        sys.exit(1)
    
    try:
        with open(COMMIT_LIST_FILE, 'r', encoding='utf-8') as f:
            commit_list = json.load(f)
    except Exception as e:
        print(f"Failed to load commit list from {COMMIT_LIST_FILE}: {str(e)}")
        sys.exit(1)
    
    for current_model in MODELS:
        print(f"\n=================== Processing with model {current_model} ===================")
        try:
            client, model_name = get_client_and_model_info(current_model)
        except Exception as e:
            print(f"Failed to initialize client for model {current_model}: {e}")
            continue
        
        total = len(commit_list)
        for idx, commit_obj in enumerate(commit_list, 1):
            commit_hash = commit_obj.get("hash")
            if not commit_hash:
                print("Warning: commit object missing 'hash' field, skipping.")
                continue
            query_entry = next(
                (item for item in similarity_data
                 if item["query_commit"]["commit_hash"] == commit_hash \
                    and item["query_commit"]["repository_name"] == repo_name),
                None
            )
            if not query_entry:
                print(f"\nCommit {commit_hash} not found in similarity data, skipping.")
                continue
            print(f"\nProcessing commit {idx}/{total}: {commit_hash} (model: {current_model})")
            optimize_function(client, repo_name, commit_hash, similarity_data, current_model, model_name)

if __name__ == "__main__":
    main()