import os
import json
import re
import glob
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from import_configs import global_config, opt_config

# 全局变量 - 测试配置
TEST_COMMITS_FILE = global_config.root_path + "python/5-opt/eval1_benchmark/test_3_commits.json"
# 存储所有被使用的Semgrep规则对应的commit的聚类文件路径
SEMGREP_FILE_IDENTIFIER = "TEST_0_85_5_False_order_commit3"

# 模式设置：控制使用哪种summary文件
SUMMARY_MODE = "cluster"  # "all" 或 "cluster"

# 筛选Semgrep扫描结果的模式：控制是否忽略同代码库其他commit对应的semgrep规则产生的结果
FILTER_SEMGREP_MODE = "False"

# 动态配置基于模式的设置
SEMGREP_RESULT_PATTERN = global_config.root_path + f"semgrep_result/{SEMGREP_FILE_IDENTIFIER}/{{repository_name}}/{{hash}}_summary_{SUMMARY_MODE}.json"

# LLM配置
MODEL_NAME = global_config.xmcp_deepseek_model
BASE_URL = global_config.xmcp_base_url
API_KEY = global_config.xmcp_api_key_unlimit
TEMPERATURE = 0.0

# 优化策略配置 - 测试设置
MAX_CODE_SEGMENTS = 3          # 只处理前3个代码段用于测试
MAX_REFERENCE_STRATEGIES = 2  # 每个代码片段使用前2个commit信息
OPTIMIZATION_ROUNDS = 2       # 每个代码片段优化2次
USE_DIFF_INFO = False         # 不使用diff信息（简化测试）
REUSE_EXISTING_RESULTS = True # 复用现有结果
MAX_WORKERS = 2               # 减少并行数用于测试

# 输出配置
OUTPUT_BASE_DIR = global_config.root_path + "knowledge_base/"

# 存储优化结果的输出文件夹名称
folder_name = f"TEST_{MODEL_NAME.replace('/', '_')}_{SEMGREP_FILE_IDENTIFIER}_{SUMMARY_MODE}_{FILTER_SEMGREP_MODE}"

def print_info(message):
    """关键信息输出 - 始终显示"""
    print(message, flush=True)

def print_verbose(message):
    """详细信息输出 - 仅在VERBOSE模式下显示"""
    if opt_config.VERBOSE:
        print(message, flush=True)

class ThreadSafeCounter:
    """线程安全的计数器"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self, amount=1):
        with self._lock:
            self._value += amount
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

def load_test_commits():
    """加载测试集commit列表"""
    try:
        with open(TEST_COMMITS_FILE, 'r', encoding='utf-8') as f:
            commits = json.load(f)
        print_info(f"Loaded {len(commits)} test commits")
        return commits
    except Exception as e:
        print_info(f"Error loading test commits: {e}")
        return None

def load_semgrep_results(commit):
    """加载单个commit的Semgrep扫描结果"""
    repo_name = commit['repository_name']
    commit_hash = commit['hash']
    
    result_path = SEMGREP_RESULT_PATTERN.format(
        repository_name=repo_name, 
        hash=commit_hash
    )
    
    if not os.path.exists(result_path):
        return None, f"Semgrep result file not found: {result_path}"
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # 直接从global_segments获取代码片段
        global_segments = result_data.get('global_segments', [])
        
        if not global_segments:
            return None, "No global_segments found"
        
        print_verbose(f"Loaded {len(global_segments)} segments")
        return global_segments, None
        
    except Exception as e:
        return None, f"Error loading semgrep results: {e}"

def find_before_func_file(commit_dir, pattern):
    """查找before_func.*文件"""
    full_pattern = os.path.join(commit_dir, pattern)
    matches = glob.glob(full_pattern)
    
    if len(matches) == 0:
        raise FileNotFoundError(f"No {pattern} file found in {commit_dir}")
    elif len(matches) > 1:
        raise ValueError(f"Multiple {pattern} files found in {commit_dir}: {matches}")
    
    return matches[0]

def validate_commit_files(commit):
    """验证commit所需文件的完整性"""
    repo_name = commit['repository_name']
    commit_hash = commit['hash']
    
    commit_dir = os.path.join(
        OUTPUT_BASE_DIR, repo_name, 
        'modified_file', commit_hash
    )
    
    if not os.path.exists(commit_dir):
        return False, f"Commit directory not found: {commit_dir}"
    
    try:
        # 检查before_func.*文件
        before_func_path = find_before_func_file(commit_dir, "before_func.*")
        
        # 检查before_func_numbered.*文件
        before_func_numbered_path = find_before_func_file(commit_dir, "before_func_numbered.*")
        
        return True, {
            'commit_dir': commit_dir,
            'before_func_path': before_func_path,
            'before_func_numbered_path': before_func_numbered_path
        }
    except Exception as e:
        return False, str(e)

def clean_output_directory(commit):
    """清理输出目录 - 精确删除特定格式的文件"""
    output_dir = os.path.join(
        OUTPUT_BASE_DIR, commit['repository_name'], 'modified_file', 
        commit['hash'], 'our', folder_name
    )
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果选择复用，则不进行任何清理
    if REUSE_EXISTING_RESULTS:
        return
    
    # 不复用时，只删除符合 {数字}_{数字}.txt 格式的文件
    try:
        pattern = os.path.join(output_dir, "*_*.txt")
        matched_files = glob.glob(pattern)
        exact_pattern = re.compile(r'^\d+_\d+\.txt$')
        
        deleted_count = 0
        for file_path in matched_files:
            filename = os.path.basename(file_path)
            if exact_pattern.match(filename):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception:
                    pass
        
        print_verbose(f"Cleaned {deleted_count} result files from {output_dir}")
        
    except Exception as e:
        print_verbose(f"Error during cleanup of {output_dir}: {e}")

def extract_segment_line_info(segment_data):
    """从新格式中提取行号信息"""
    in_func_lines = segment_data.get('in_func_file_lines', {})
    if isinstance(in_func_lines, dict):
        return {
            'line_start': in_func_lines.get('start', 0),
            'line_end': in_func_lines.get('end', 0)
        }
    
    # 兼容旧格式
    return {
        'line_start': segment_data.get('line_start', 0),
        'line_end': segment_data.get('line_end', 0)
    }

def extract_reference_strategies(segment_data, max_strategies):
    """提取参考优化策略"""
    messages = segment_data.get('messages', [])
    source_commits = segment_data.get('source_commits', [])
    
    # 确保messages和source_commits数量一致
    min_count = min(len(messages), len(source_commits))
    strategies = []
    
    for i in range(min(min_count, max_strategies)):
        message = messages[i] if i < len(messages) else "No optimization message"
        source_commit = source_commits[i] if i < len(source_commits) else {}
        
        strategy = {
            'message': message,
            'source_commit': source_commit,
            'diff': None  # 测试时不使用diff
        }
        strategies.append(strategy)
    
    return strategies

def build_strategies_section(reference_strategies):
    """构建参考策略部分的文本"""
    if not reference_strategies:
        return "No reference strategies available."
    
    strategies_text = ""
    for i, strategy in enumerate(reference_strategies, 1):
        strategies_text += f"Strategy {i}:\n"
        strategies_text += f"Optimization suggestion: {strategy['message']}\n"
        
        source_commit = strategy['source_commit']
        if source_commit:
            repo = source_commit.get('repository_name', 'unknown')
            hash_short = source_commit.get('hash', 'unknown')[:8]
            strategies_text += f"Source: {repo}:{hash_short}\n"
        
        strategies_text += "\n"
    
    return strategies_text.strip()

def build_optimization_prompt(before_func_numbered, segment_info, reference_strategies, before_func):
    """构建优化prompt"""
    strategies_section = build_strategies_section(reference_strategies)
    
    prompt = f"""Please optimize the following C/C++ function based on the identified code segment and reference optimization strategies.

FUNCTION WITH LINE NUMBERS:
```cpp
{before_func_numbered}
```

TARGET CODE SEGMENT TO OPTIMIZE:
Lines {segment_info['line_start']} to {segment_info['line_end']}

REFERENCE OPTIMIZATION STRATEGIES:
Note: The following strategies may contain duplicates or overlapping suggestions.

{strategies_section}

ORIGINAL FUNCTION TO OPTIMIZE:
```cpp
{before_func}
```

REQUIREMENTS:
1. First, identify the specific code segment that needs optimization based on the given line numbers ({segment_info['line_start']}-{segment_info['line_end']})
2. Carefully analyze and understand each reference optimization strategy provided
3. Determine if each strategy is applicable to the identified target code segment
4. Verify that applying each strategy will preserve the original semantics and correctness
5. Only apply strategies that are both applicable and semantically safe
6. Maintain exact functionality throughout the entire function
7. Your response must contain exactly ONE code block with the complete optimized function. You may provide explanations and analysis in your response, but ensure there is ONLY ONE code block containing the entire optimized function."""
    
    return prompt

def get_llm_client():
    """创建LLM客户端"""
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)

def call_llm_for_optimization(client, prompt):
    """调用LLM进行优化"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

def extract_code_from_response(response):
    """从LLM响应中提取代码"""
    # 匹配代码块
    code_block_pattern = r'```(?:cpp|c|C\+\+)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    
    if matches:
        # 取第一个代码块
        return matches[0].strip()
    else:
        # 如果没有找到代码块，返回原始内容
        return response.strip()

def build_output_path(commit, segment_id, round_num):
    """构建单个优化结果的输出路径"""
    output_dir = os.path.join(
        OUTPUT_BASE_DIR, commit['repository_name'], 'modified_file', 
        commit['hash'], 'our', folder_name
    )
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{segment_id}_{round_num}.txt")

def save_optimization_result(code, output_path):
    """保存优化结果"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        return True
    except Exception as e:
        print_verbose(f"Error saving result to {output_path}: {e}")
        return False

def should_skip_task(output_path, reuse_existing):
    """检查是否应跳过当前任务"""
    if not reuse_existing:
        return False
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    
    return False

def process_single_optimization_task(task):
    """处理单个优化任务"""
    commit = task['commit']
    segment_id = task['segment_id']
    round_num = task['round_num']
    segment_data = task['segment_data']
    file_paths = task['file_paths']
    output_path = task['output_path']
    
    try:
        # 检查是否跳过
        if should_skip_task(output_path, REUSE_EXISTING_RESULTS):
            return {'status': 'skipped', 'task_id': f"{commit['hash'][:8]}:{segment_id}:{round_num}"}
        
        # 读取文件内容
        with open(file_paths['before_func_path'], 'r', encoding='utf-8') as f:
            before_func = f.read().strip()
        
        with open(file_paths['before_func_numbered_path'], 'r', encoding='utf-8') as f:
            before_func_numbered = f.read().strip()
        
        # 提取参考策略
        reference_strategies = extract_reference_strategies(segment_data, MAX_REFERENCE_STRATEGIES)
        
        # 从新格式中提取行号信息
        segment_info = extract_segment_line_info(segment_data)
        
        # 验证行号信息
        if segment_info['line_start'] <= 0 or segment_info['line_end'] <= 0:
            return {'status': 'failed', 'task_id': f"{commit['hash'][:8]}:{segment_id}:{round_num}", 'error': 'invalid_line_numbers'}
        
        # 构建prompt
        prompt = build_optimization_prompt(
            before_func_numbered, segment_info, reference_strategies, before_func
        )
        
        # 调用LLM
        client = get_llm_client()
        response = call_llm_for_optimization(client, prompt)
        
        # 提取代码
        optimized_code = extract_code_from_response(response)
        
        # 保存结果
        success = save_optimization_result(optimized_code, output_path)
        
        if success:
            return {'status': 'success', 'task_id': f"{commit['hash'][:8]}:{segment_id}:{round_num}"}
        else:
            return {'status': 'failed', 'task_id': f"{commit['hash'][:8]}:{segment_id}:{round_num}", 'error': 'save_failed'}
    
    except Exception as e:
        return {'status': 'failed', 'task_id': f"{commit['hash'][:8]}:{segment_id}:{round_num}", 'error': str(e)}

def generate_optimization_tasks(commits):
    """生成所有优化任务"""
    tasks = []
    cleaned_commits = set()  # 避免重复清理
    
    for commit in commits:
        # 验证文件
        valid, file_info = validate_commit_files(commit)
        if not valid:
            print_verbose(f"Skipping commit {commit['hash'][:8]}: {file_info}")
            continue
        
        # 加载Semgrep结果
        segments, error = load_semgrep_results(commit)
        if not segments:
            print_verbose(f"Skipping commit {commit['hash'][:8]}: {error}")
            continue
        
        # 首次处理该commit时清理目录
        commit_key = f"{commit['repository_name']}:{commit['hash']}"
        if commit_key not in cleaned_commits:
            clean_output_directory(commit)
            cleaned_commits.add(commit_key)
        
        # 取前N个代码片段
        selected_segments = segments[:MAX_CODE_SEGMENTS]
        
        # 为每个片段生成多轮任务
        for segment_id, segment_data in enumerate(selected_segments, 1):
            for round_num in range(1, OPTIMIZATION_ROUNDS + 1):
                output_path = build_output_path(commit, segment_id, round_num)
                
                task = {
                    'commit': commit,
                    'segment_id': segment_id,
                    'round_num': round_num,
                    'segment_data': segment_data,
                    'file_paths': file_info,
                    'output_path': output_path
                }
                tasks.append(task)
    
    return tasks

def validate_configuration():
    """验证配置的有效性"""
    if SUMMARY_MODE not in ["all", "cluster"]:
        raise ValueError(f"Invalid SUMMARY_MODE: {SUMMARY_MODE}. Must be 'all' or 'cluster'")
    
    if MAX_CODE_SEGMENTS <= 0:
        raise ValueError(f"MAX_CODE_SEGMENTS must be positive, got: {MAX_CODE_SEGMENTS}")
    
    if OPTIMIZATION_ROUNDS <= 0:
        raise ValueError(f"OPTIMIZATION_ROUNDS must be positive, got: {OPTIMIZATION_ROUNDS}")
    
    if MAX_REFERENCE_STRATEGIES <= 0:
        raise ValueError(f"MAX_REFERENCE_STRATEGIES must be positive, got: {MAX_REFERENCE_STRATEGIES}")

def main():
    """主函数"""
    print_info("=== Semgrep-Based Code Optimization Started (TEST MODE) ===")
    
    # 验证配置
    try:
        validate_configuration()
    except ValueError as e:
        print_info(f"Configuration error: {e}")
        return
    
    # 加载测试commit
    print_info("Loading test commits...")
    commits = load_test_commits()
    if not commits:
        print_info("Failed to load test commits")
        return
    
    # 生成优化任务
    print_info("Generating optimization tasks...")
    tasks = generate_optimization_tasks(commits)
    
    if not tasks:
        print_info("No optimization tasks generated")
        return
    
    print_info(f"Generated {len(tasks)} optimization tasks")
    print_info(f"Configuration: Mode={SUMMARY_MODE.upper()}, Model={MODEL_NAME}, Segments={MAX_CODE_SEGMENTS}, Rounds={OPTIMIZATION_ROUNDS}")
    print_info(f"Reference strategies={MAX_REFERENCE_STRATEGIES}, Use diff={USE_DIFF_INFO}")
    print_info(f"Reuse existing={REUSE_EXISTING_RESULTS}, Workers={MAX_WORKERS}")
    print_info(f"Output folder: {folder_name}")
    
    # 统计变量
    success_counter = ThreadSafeCounter()
    failed_counter = ThreadSafeCounter()
    skipped_counter = ThreadSafeCounter()
    
    # 并行处理任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {}
        
        for task in tasks:
            future = executor.submit(process_single_optimization_task, task)
            future_to_task[future] = task
        
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc="Processing optimization tasks", unit="task") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    
                    if result['status'] == 'success':
                        success_counter.increment()
                    elif result['status'] == 'skipped':
                        skipped_counter.increment()
                    else:  # failed
                        failed_counter.increment()
                        print_verbose(f"Task failed {result['task_id']}: {result.get('error', 'unknown')}")
                    
                    pbar.set_postfix({
                        'Success': success_counter.value,
                        'Failed': failed_counter.value,
                        'Skipped': skipped_counter.value,
                        'Current': result['task_id']
                    })
                    
                except Exception as e:
                    failed_counter.increment()
                    task_id = f"{task['commit']['hash'][:8]}:{task['segment_id']}:{task['round_num']}"
                    print_verbose(f"Task exception {task_id}: {e}")
                
                finally:
                    pbar.update(1)
    
    # 输出最终统计
    print_info(f"\n=== Final Statistics ===")
    print_info(f"Total tasks: {len(tasks)}")
    print_info(f"Successfully processed: {success_counter.value}")
    print_info(f"Skipped (existing): {skipped_counter.value}")
    print_info(f"Failed: {failed_counter.value}")
    print_info(f"Success rate: {success_counter.value/len(tasks)*100:.1f}%")

if __name__ == "__main__":
    main()

