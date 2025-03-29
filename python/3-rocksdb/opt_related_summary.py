import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import OpenAI

# 全局变量：设置是否在 after_func_{CURRENT_MODEL}.txt 已存在时重新生成优化结果
# 设置为 True 时，即使文件存在，也会重新调用 LLM 生成优化结果
# 设置为 False 时，如果文件已存在，则跳过该提交的优化过程
REGENERATE_EXISTING = True

# prompt 中的例子数量
MAX_EXAMPLES = 4

# 定义需要处理的模型
MODELS = ['gpt-4o']
# 支持的模型列表
SUPPORTED_MODELS = ['deepseek-v3', 'deepseek-reasoner', 'gpt-4o', 'o1-mini', 'o3-mini']

# 放 commit 相关文件的路径
COMMIT_DIR = config.root_path + "benchmark/{repo}/modified_file/{commit}"
# 放结果文件的子文件夹名
OUTPUT_DIRNAME = "langchain+cosine"
# 记录llm优化结果的文件名
OUTPUT_FILENAME = f"related_{MAX_EXAMPLES}_summary" + "_{model}.txt"

# commit列表文件，文件内包含包含 "hash" 字段的commit对象数组
COMMIT_LIST_FILE = os.path.join(config.root_path, 'benchmark', 'rocksdb', 'filtered_test_result.json')

# 加载全局的相似性数据
similarity_path = os.path.join(config.root_path, "commit_similarity_langchain+cosine.json")

def get_client_and_model_info(model):
    """
    根据传入的模型名称初始化客户端，返回(client, model_name)。
    不同模型可能需要不同的API参数。
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Error: Unsupported model '{model}'. Supported: {SUPPORTED_MODELS}")

    if model == 'deepseek-v3':
        base_url = config.deepseek_base_url
        api_key = config.deepseek_api_key
        model_name = 'deepseek-chat'
    elif model == 'deepseek-reasoner':
        base_url = config.deepseek_base_url
        api_key = config.deepseek_api_key
        model_name = 'deepseek-reasoner'
    else:
        base_url = config.closeai_base_url
        api_key = config.closeai_api_key
        model_name = model

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    except Exception as e:
        raise RuntimeError(f"模型 {model} 客户端初始化失败：{e}")
    
    return client, model_name

def load_summary_text(repo_name, commit_hash):
    """
    加载指定commit的summary.txt文件。
    文件路径：root_path/result/<repository_name>/modified_file/<commit_hash>/summary.txt
    返回文件内容字符串；若加载失败则返回None。
    """
    commit_dir = os.path.join(config.root_path, 'result', repo_name, 'modified_file', commit_hash)
    summary_file = os.path.join(commit_dir, 'summary.txt')
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"加载commit {commit_hash}的summary.txt失败：{e}")
        return None

def load_target_function(repo_name, commit_hash):
    """
    从before_func.txt加载目标函数（待优化函数）。
    文件路径：root_path/benchmark/<repository_name>/modified_file/<commit_hash>/before_func.txt
    返回函数代码字符串；若加载失败返回None。
    """
    commit_dir = os.path.join(config.root_path, 'benchmark', repo_name, 'modified_file', commit_hash)
    before_file = os.path.join(commit_dir, 'before_func.txt')
    try:
        with open(before_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"加载commit {commit_hash}的before_func.txt失败：{e}")
        return None

def build_similar_commits_section(query_commit_hash, repo_name, similarity_data):
    """
    构造相似commit信息部分文本，包含前MAX_EXAMPLES个相似commit的summary.txt内容。
    similarity_data为列表，每个元素包含：
      - query_commit: 包含 "commit_hash" 和 "repository_name" 的字典；
      - similar_commits: 相似commit对象列表，每个对象应包含 "commit_hash"（或 "hash"）、可选 "repository_name"（默认为repo_name）及 "similarity_score"。
    处理流程：
      1. 首先从相似性数据中获得与query_commit匹配的记录；
      2. 从该记录中选择相似度最高的MAX_EXAMPLES个commit作为参考项；
      3. 将这n个commit按照相似度从低到高排序后，将对应的summary.txt依次拼接后返回。
    """
    query_entry = next(
        (item for item in similarity_data 
         if item["query_commit"]["commit_hash"] == query_commit_hash 
         and item["query_commit"]["repository_name"] == repo_name),
        None
    )
    if not query_entry:
        return "No similar commits found."

    # 先按相似度从高到低选择top n相似commit
    top_commits = sorted(query_entry["similar_commits"],
                         key=lambda x: x["similarity_score"],
                         reverse=True)[:MAX_EXAMPLES]
    # 然后将这n个commit按照相似度从低到高排序
    selected_commits = sorted(top_commits, key=lambda x: x["similarity_score"])

    similar_texts = []
    for idx, sim_commit in enumerate(selected_commits, start=1):
        sim_commit_hash = sim_commit.get("commit_hash") or sim_commit.get("hash")
        sim_repo = sim_commit.get("repository_name", repo_name)
        similarity_score = sim_commit.get("similarity_score", 0)
        sim_summary = load_summary_text(sim_repo, sim_commit_hash)
        if sim_summary is not None:
            similar_texts.append(
                f"==== Similar Commit {idx} (Similarity: {similarity_score:.1f}) ====\n{sim_summary}"
            )
    if similar_texts:
        return "\n\n".join(similar_texts)
    else:
        return "No summary information available for similar commits."

def generate_prompt(target_function, similar_section):
    """
    构造给LLM的提示信息（prompt），提示信息全部为英文，包含以下部分：
      1. 开头介绍：说明我们的目的是优化一个函数，并将随后提供若干个相似的例子。每个例子来自真实代码库的commit，这些例子中，
         首先使用我们的分类系统对commit进行分类并给出分类结果，然后从三个角度提供一个简要summary。
      2. 具体的分类系统：（省略）
      3. 接下来依次给出所有的例子（相似commit的summary信息），以及需要被优化的目标函数代码。

    返回完整prompt字符串。
    """
    introduction = (
        "The following prompt is composed of several parts:\n"
        "---------------"
        "# Introduction \n"
        "Our goal is to optimize a function. "
        "We provide several similar examples obtained from real repository commits. "
        "For each example, a classification of the commit is performed using our specific classification system, "
        "followed by concise summaries from three perspectives.\n\n"
        "---------------"
        "# Classification System \n"
        "   1. Causes of Performance Defects:\n"
        "      - Question 1: Which of the following major categories does the problem addressed by the optimization belong to?\n"
        "          • Structural\n"
        "          • Misunderstanding/Misuse\n"
        "          • Missing Operation\n"
        "          • Redundancy or Memoization Opportunity\n"
        "          • Others\n"
        "      - If 'Structural' is chosen, Question 2.1: Which specific type of structural issue does the optimization involve?\n"
        "          • Condition: Optimized inappropriate or incomplete conditional statements.\n"
        "          • Loop: Optimized inefficient looping constructs.\n"
        "          • Data Structure: Optimized the selection or use of data structures.\n"
        "          • Synchronization: Optimized the usage of synchronization mechanisms.\n"
        "          • Query: Optimized inefficient query statements.\n"
        "      - If 'Misunderstanding/Misuse' is chosen, Question 2.2: Which specific type of misunderstanding or misuse does the optimization involve?\n"
        "          • Call to API Method: Optimized the use of inappropriate API functions or parameters.\n"
        "          • Inter-procedural: Optimized performance issues exposed when independently implemented methods are used together.\n"
        "          • Unnecessary Computation: Optimized unnecessary computation tasks.\n"
        "          • Configuration: Optimized mismatches between program configurations and user performance expectations.\n"
        "      - For 'Missing Operation', 'Redundancy or Memoization Opportunity' or 'Others': No further classification is needed.\n\n"
        "   2. Impact of the Performance Defect:\n"
        "      - Question 2: Which of the following aspects is primarily improved by the optimization?\n"
        "          • Execution Time: The optimization reduces the program’s execution time.\n"
        "          • Memory Utilization: The optimization reduces memory consumption or prevents memory leaks.\n"
        "          • Energy Consumption: The optimization reduces the device’s energy consumption.\n"
        "          • No Significant Improvement: The optimization does not significantly improve any of the above aspects.\n\n"
        "   3. Scope of Application:\n"
        "      - Question 3: What type of project does the optimization pertain to?\n"
        "          • Server: The optimization involves server-related code.\n"
        "          • Web: The optimization involves browser or web application related code.\n"
        "          • Graphical User Interface (GUI): The optimization involves GUI-related code.\n"
        "          • Database: The optimization involves database-related code.\n"
        "          • Communication: The optimization involves communication-related code.\n"
        "          • Multiple Processes: The optimization involves code for concurrent, parallel, or asynchronous tasks.\n"
        "          • System Software: The optimization involves code related to operating systems, compilers, etc.\n"
        "          • Others: Other specific project types.\n"
        "      - Question 4: How general is the optimization?\n"
        "          • General Optimization: Applicable to most programming languages and programs.\n"
        "          • Language-specific Optimization: Relates to features of a specific programming language.\n"
        "              - If 'Language-specific Optimization' is chosen, Question 4.1: Which programming language does the optimization involve?\n"
        "                  • Java\n"
        "                  • C++\n"
        "                  • Python\n"
        "                  • Others (please specify)\n\n"
    )

    additional_instructions = (
        "---------------"
        "# Summary Information from Similar Commits \n"
        f"{similar_section}\n\n"
    )

    prompt = (
        f"{introduction}"
        f"{additional_instructions}\n\n"
        "Using the above similar commits’ information as a reference, please optimize the following function.\n"
        "---------------"
        "# Function to Optimize \n"
        f"```cpp\n{target_function}\n```\n\n"
        "# Requirements\n"
        "   1. The optimized function must maintain EXACT functionality.\n"
        "   2. Return ONLY the complete optimized function code without any explanations or comments.\n\n"
        "Please return the optimized function code in plain text."
    )
    # print(prompt)
    return prompt

def optimize_function(client, repo_name, commit_hash, current_model, model_name, similarity_data):
    """
    对于每个commit：
      1. 从before_func.txt加载目标函数；
      2. 根据similarity_data构造包含最相似commit的summary.txt信息部分文本；
      3. 构造完整prompt（包括介绍、分类系统、相似commit例子以及待优化的函数代码）；
      4. 调用LLM优化目标函数；
      5. 将优化结果保存到文件（格式为 related_{MAX_EXAMPLES}_summary_{current_model}.txt）。
    """
    # 对于待优化的 commit，其目录位于 benchmark 下
    commit_dir = COMMIT_DIR.format(repo = repo_name, commit = commit_hash)
    
    # 创建输出目录
    output_dir = commit_dir + "/" + OUTPUT_DIRNAME
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径，文件名包含当前模型标识
    output_path = os.path.join(output_dir, OUTPUT_FILENAME.format(model = current_model))
    
    if os.path.exists(output_path) and not REGENERATE_EXISTING:
        print(f"commit {commit_hash} 已存在 {output_filename} 文件，跳过。")
        return
    
    # 加载目标函数代码
    target_function = load_target_function(repo_name, commit_hash)
    if not target_function:
        print(f"commit {commit_hash} 缺少目标函数，跳过。")
        return
    
    # 构造相似commit信息部分
    similar_section = build_similar_commits_section(commit_hash, repo_name, similarity_data)
    
    # 根据目标函数和相似commit信息构造完整prompt（提示信息部分全部为英文）
    full_prompt = generate_prompt(target_function, similar_section)
    
    try:
        # 调用LLM生成优化后的函数代码
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
        
        optimized_code = response.choices[0].message.content.strip()
        # 移除可能的Markdown格式标记
        lines = optimized_code.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip().startswith("```"):
            optimized_code = "\n".join(lines[1:-1]).strip()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(optimized_code)
        status = "Regenerated" if os.path.exists(output_path) else "Created"
        print(f"{status} 优化结果：commit {commit_hash} (模型: {current_model})，文件：{output_path}")
    except Exception as e:
        print(f"commit {commit_hash} 调用API失败 (模型: {current_model})：{e}")

def main():
    # 加载commit列表
    try:
        with open(COMMIT_LIST_FILE, 'r', encoding='utf-8') as f:
            commit_list = json.load(f)
    except Exception as e:
        print(f"加载commit列表 {COMMIT_LIST_FILE} 失败：{e}")
        sys.exit(1)
    
    try:
        with open(similarity_path, 'r', encoding='utf-8') as f:
            similarity_data = json.load(f)
    except Exception as e:
        print(f"加载相似性数据 {similarity_path} 失败：{e}")
        sys.exit(1)
    
    # 遍历每个模型进行处理，此处使用MODELS列表中所有模型
    for current_model in MODELS:
        print(f"\n=================== 使用模型 {current_model} 处理 ====================")
        try:
            client, model_name = get_client_and_model_info(current_model)
        except Exception as e:
            print(f"初始化模型 {current_model} 客户端失败：{e}")
            continue
        
        total = len(commit_list)
        for idx, commit_obj in enumerate(commit_list, 1):
            commit_hash = commit_obj.get("hash")
            if not commit_hash:
                print("警告：commit对象缺少 'hash' 字段，跳过。")
                continue
            # 默认仓库名称为 "rocksdb"
            repo_name = commit_obj.get("repository_name", "rocksdb")
            print(f"\n正在处理 commit {idx}/{total}：{commit_hash} (模型: {current_model})")
            optimize_function(client, repo_name, commit_hash, current_model, model_name, similarity_data)

if __name__ == "__main__":
    main()