import os
import json
import threading
import contextlib
import sys
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from import_configs import global_config, generate_config
from process_once import generate_semgrep_rule_from_file

def print_info(message):
    """关键信息输出 - 始终显示"""
    print(message)

def print_verbose(message):
    """详细信息输出 - 仅在VERBOSE模式下显示"""
    if generate_config.VERBOSE:
        print(message)

@contextlib.contextmanager
def suppress_stdout_stderr():
    """抑制stdout和stderr输出"""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def find_before_file(commit_dir_path):
    """查找before文件（任意后缀）"""
    pattern = os.path.join(commit_dir_path, "before.*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]  # 返回第一个匹配的文件
    return None

def build_commit_paths(commit_dir_path):
    """构建commit相关的文件路径"""
    commit_num = os.path.basename(commit_dir_path.rstrip('/'))
    
    # 查找before文件
    before_path = find_before_file(commit_dir_path)
    
    return {
        'diff_file': os.path.join(commit_dir_path, 'diff.txt'),
        'before': before_path,
        'semgrep_dir': os.path.join(commit_dir_path, 'semgrep'),
        'commit_num': commit_num
    }

def validate_commit_directory(commit_paths):
    """验证commit目录和必要文件"""
    if not os.path.exists(commit_paths['diff_file']):
        return False, f"Diff file not found: {commit_paths['diff_file']}"
    
    if not commit_paths['before'] or not os.path.exists(commit_paths['before']):
        return False, f"Before file not found in directory"
    
    # 检查diff文件是否为空
    try:
        with open(commit_paths['diff_file'], 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return False, f"Diff file is empty: {commit_paths['diff_file']}"
            
            print_verbose(f"✅ Diff file validated: {commit_paths['diff_file']}")
            print_verbose(f"   Content preview (first 200 chars): {content[:200]}...")
    except Exception as e:
        return False, f"Error reading diff file: {e}"
    
    print_verbose(f"✅ Before file validated: {commit_paths['before']}")
    
    return True, None

def should_continue_fixing(json_result, error_msg):
    """判断是否需要继续修复迭代（复用process_once.py的逻辑）"""
    # 如果有执行错误，需要继续修复
    if error_msg:
        return True
    
    # 如果没有JSON结果或没有errors，不需要修复
    if not json_result or not json_result.get("errors"):
        return False
    
    # 检查每个错误的级别
    for error in json_result.get("errors", []):
        level = error.get("level", "").lower()
        # 如果有严重错误，需要继续修复
        if level in ["fatal", "error"]:
            return True
    
    # 所有错误都是警告级别或更低，不需要继续修复
    return False

def check_existing_rules(semgrep_dir, generation_count, regenerate_on_json_error=True):
    """
    检查已存在的规则文件（同时检查.yaml和.yml扩展名，并验证内容有效性和JSON结果）
    
    Args:
        semgrep_dir: semgrep规则目录路径
        generation_count: 需要生成的规则数量
        regenerate_on_json_error: 如果规则已存在但JSON结果中有fatal/error级别的错误，是否重新生成
                                 默认True，表示会重新生成有错误的规则
    
    Returns:
        tuple: (existing_files, missing_files) 已存在的规则编号列表和缺失的规则编号列表
    """
    existing_files = []
    missing_files = []
    
    if not os.path.exists(semgrep_dir):
        missing_files = list(range(1, generation_count + 1))
        return existing_files, missing_files
    
    for i in range(1, generation_count + 1):
        yaml_path = os.path.join(semgrep_dir, f"{i}.yaml")
        yml_path = os.path.join(semgrep_dir, f"{i}.yml")
        json_path = os.path.join(semgrep_dir, f"{i}.json")
        
        # 检查是否存在有效的规则文件（.yaml 或 .yml）
        rule_file = None
        if os.path.exists(yaml_path) and os.path.getsize(yaml_path) > 0:
            rule_file = yaml_path
        elif os.path.exists(yml_path) and os.path.getsize(yml_path) > 0:
            rule_file = yml_path
        
        # 如果YAML文件存在，验证内容是否包含基本的semgrep规则关键字
        if rule_file:
            try:
                with open(rule_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content or ('rules:' not in content and 'rule:' not in content):
                        # 文件存在但内容无效，视为缺失
                        missing_files.append(i)
                        continue
            except:
                # 读取文件失败，视为缺失
                missing_files.append(i)
                continue
            
            # YAML文件有效，检查JSON结果文件
            # 如果JSON文件不存在，认为规则还未运行，需要重新生成
            if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
                print_verbose(f"Rule {i}: YAML exists but JSON missing, will regenerate")
                missing_files.append(i)
                continue
            
            # JSON文件存在，根据配置决定是否检查错误
            if regenerate_on_json_error:
                # 检查是否有fatal/error级别的错误
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_result = json.load(f)
                    
                    # 如果JSON结果中有fatal/error级别的错误，需要重新生成
                    if should_continue_fixing(json_result, None):
                        print_verbose(f"Rule {i}: YAML exists but JSON has fatal/error, will regenerate")
                        missing_files.append(i)
                        continue
                    
                    # YAML和JSON都存在且有效，可以复用
                    existing_files.append(i)
                except Exception as e:
                    # JSON文件解析失败，视为缺失，需要重新生成
                    print_verbose(f"Rule {i}: JSON parse error ({str(e)}), will regenerate")
                    missing_files.append(i)
            else:
                # 不检查JSON错误，只要YAML和JSON文件都存在就认为可以复用
                existing_files.append(i)
        else:
            # YAML文件不存在，需要生成
            missing_files.append(i)
    
    return existing_files, missing_files

def setup_semgrep_directory(semgrep_dir, reuse_existing):
    """设置semgrep目录"""
    if not reuse_existing and os.path.exists(semgrep_dir):
        try:
            import shutil
            shutil.rmtree(semgrep_dir)
            print_verbose(f"Removed existing semgrep directory: {semgrep_dir}")
        except Exception as e:
            return False, f"Failed to remove existing semgrep directory: {e}"
    
    try:
        os.makedirs(semgrep_dir, exist_ok=True)
        print_verbose(f"✅ Semgrep directory ready: {semgrep_dir}")
        return True, None
    except Exception as e:
        return False, f"Failed to create semgrep directory: {e}"

def create_generation_task(rule_number, commit_paths):
    """创建单个规则生成任务"""
    return {
        'rule_number': rule_number,
        'yaml_path': os.path.join(commit_paths['semgrep_dir'], f"{rule_number}.yaml"),
        'json_path': os.path.join(commit_paths['semgrep_dir'], f"{rule_number}.json")
    }

def generate_single_rule_safe(task, commit_paths, config_params):
    """线程安全的单规则生成函数"""
    print_verbose(f"\n{'='*50}")
    print_verbose(f"GENERATION ATTEMPT #{task['rule_number']}")
    print_verbose(f"{'='*50}")
    
    try:
        if generate_config.VERBOSE:
            # 正常输出所有信息
            success, error_msg = generate_semgrep_rule_from_file(
                diff_file_path=commit_paths['diff_file'],
                before_path=commit_paths['before'],
                target_yaml_path=task['yaml_path'],
                commit_num=commit_paths['commit_num'],
                max_round=config_params['max_round'],
                use_langsmith=config_params['use_langsmith']
            )
        else:
            # 抑制详细输出，只保留关键信息
            with suppress_stdout_stderr():
                success, error_msg = generate_semgrep_rule_from_file(
                    diff_file_path=commit_paths['diff_file'],
                    before_path=commit_paths['before'],
                    target_yaml_path=task['yaml_path'],
                    commit_num=commit_paths['commit_num'],
                    max_round=config_params['max_round'],
                    use_langsmith=config_params['use_langsmith']
                )
        
        if success:
            print_verbose(f"✅ Successfully generated rule #{task['rule_number']}: {task['yaml_path']}")
            return True, None
        else:
            print_verbose(f"❌ Failed to generate rule #{task['rule_number']}: {error_msg}")
            return False, error_msg
            
    except Exception as e:
        print_verbose(f"❌ Error during generation #{task['rule_number']}: {str(e)}")
        return False, str(e)

def execute_parallel_generation(tasks, commit_paths, config_params, max_workers=None):
    """执行并行规则生成"""
    if max_workers is None:
        max_workers = generate_config.COMMIT_MAX_WORKERS
    
    results = {}
    
    if not tasks:
        return results
    
    print_verbose(f"\n{'='*60}")
    print_verbose(f"STARTING PARALLEL GENERATION")
    print_verbose(f"{'='*60}")
    print_verbose(f"Tasks: {len(tasks)}")
    print_verbose(f"Max workers: {max_workers}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {}
        for task in tasks:
            future = executor.submit(generate_single_rule_safe, task, commit_paths, config_params)
            future_to_task[future] = task
        
        # 收集结果，显示进度
        desc = "Generating rules"
        with tqdm(total=len(tasks), desc=desc, unit="rule", 
                  disable=False, leave=True, position=0) as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    success, error_msg = future.result()
                    results[task['rule_number']] = (success, error_msg)
                except Exception as e:
                    results[task['rule_number']] = (False, str(e))
                finally:
                    pbar.update(1)
    
    return results

def print_generation_summary(results, generation_count, existing_count=0):
    """打印生成结果摘要"""
    success_count = sum(1 for success, _ in results.values() if success)
    total_generated = len(results)
    
    print_info(f"\n{'='*50}")
    print_info(f"GENERATION SUMMARY")
    print_info(f"{'='*50}")
    print_info(f"Total rules needed: {generation_count}")
    print_info(f"Existing rules reused: {existing_count}")
    print_info(f"New rules attempted: {total_generated}")
    print_info(f"New rules successful: {success_count}")
    print_info(f"Final success rate: {(existing_count + success_count) / generation_count * 100:.1f}%")
    
    # 显示失败的规则编号
    failed_rules = [num for num, (success, _) in results.items() if not success]
    if failed_rules:
        print_info(f"Failed rules: {sorted(failed_rules)}")
        
        print_verbose(f"\nDetailed failure reasons:")
        for num in sorted(failed_rules):
            _, error_msg = results[num]
            print_verbose(f"  Rule {num}: {error_msg}")

def process_commit_semgrep_rules(commit_dir_path, generation_count=None, use_langsmith=None, 
                               reuse_existing=None, max_round=None, regenerate_on_json_error=None):
    """
    处理单个commit的Semgrep规则生成
    
    Args:
        commit_dir_path: commit文件夹路径
        generation_count: 需要生成的规则数量，默认使用config配置
        use_langsmith: 是否使用LangSmith，默认使用config配置
        reuse_existing: 是否复用已有结果，默认使用config配置
        max_round: 最大迭代轮数，默认使用config配置
        regenerate_on_json_error: 如果规则已存在但JSON结果中有fatal/error级别的错误，是否重新生成
                                 默认使用config配置
    
    Returns:
        int: 成功生成的规则数量
    """
    # 使用config默认值
    if generation_count is None:
        generation_count = generate_config.COMMIT_GENERATION_COUNT
    if use_langsmith is None:
        use_langsmith = generate_config.LLM_USE_LANGSMITH
    if reuse_existing is None:
        reuse_existing = generate_config.COMMIT_REUSE_EXISTING
    if max_round is None:
        max_round = generate_config.LLM_MAX_GENERATION_ROUNDS
    if regenerate_on_json_error is None:
        regenerate_on_json_error = generate_config.COMMIT_REGENERATE_ON_JSON_ERROR
    
    print_info(f"{'='*60}")
    print_info(f"PROCESSING COMMIT: {os.path.basename(commit_dir_path)}")
    print_info(f"{'='*60}")
    
    # 构建文件路径
    commit_paths = build_commit_paths(commit_dir_path)
    
    # 验证输入
    valid, error_msg = validate_commit_directory(commit_paths)
    if not valid:
        print_info(f"❌ Error: {error_msg}")
        return 0
    
    # 设置semgrep目录
    success, error_msg = setup_semgrep_directory(commit_paths['semgrep_dir'], reuse_existing)
    if not success:
        print_info(f"❌ Error: {error_msg}")
        return 0
    
    # 检查现有规则
    existing_files, missing_files = check_existing_rules(
        commit_paths['semgrep_dir'], 
        generation_count, 
        regenerate_on_json_error=regenerate_on_json_error
    )
    
    print_info(f"Target: {generation_count} rules")
    print_info(f"Existing: {len(existing_files)} rules")
    print_info(f"Missing: {len(missing_files)} rules")
    
    if not missing_files:
        print_info("✅ All rules already exist. Skipping generation.")
        return len(existing_files)
    
    # 配置参数
    config_params = {
        'max_round': max_round,
        'use_langsmith': use_langsmith
    }
    
    # 创建生成任务
    tasks = [create_generation_task(rule_num, commit_paths) for rule_num in missing_files]
    
    # 执行并行生成
    if not generate_config.VERBOSE:
        print_info(f"Starting generation of {len(tasks)} rules... (use VERBOSE=True to see details)")
    
    results = execute_parallel_generation(tasks, commit_paths, config_params)
    
    # 打印摘要
    print_generation_summary(results, generation_count, len(existing_files))
    
    # 返回总成功数量
    success_new = sum(1 for success, _ in results.values() if success)
    return len(existing_files) + success_new

def main():
    """主函数：使用默认参数处理测试案例"""
    # 使用config中的默认值
    GENERATION_COUNT = generate_config.COMMIT_GENERATION_COUNT
    USE_LANGSMITH = generate_config.LLM_USE_LANGSMITH
    REUSE_EXISTING = generate_config.COMMIT_REUSE_EXISTING
    MAX_ROUND = generate_config.LLM_MAX_GENERATION_ROUNDS
    
    # 测试路径仍在脚本中定义
    COMMIT_DIR_PATH = global_config.root_path + "knowledge_base/tauri-apps_tauri/modified_file/836ee60dc8821a84fda66c72d66fdeccce6e988b/"
    
    print_info(f"{'='*60}")
    print_info(f"PROCESS_COMMIT.PY - MAIN FUNCTION")
    print_info(f"{'='*60}")
    print_verbose(f"Commit directory: {COMMIT_DIR_PATH}")
    print_verbose(f"Generation count: {GENERATION_COUNT}")
    print_verbose(f"Use LangSmith: {USE_LANGSMITH}")
    print_verbose(f"Reuse existing: {REUSE_EXISTING}")
    print_verbose(f"Max rounds: {MAX_ROUND}")
    print_verbose(f"Verbose mode: {generate_config.VERBOSE}")
    
    # 验证commit目录存在
    if not os.path.exists(COMMIT_DIR_PATH):
        print_info(f"❌ Error: Commit directory not found: {COMMIT_DIR_PATH}")
        return
    
    # 执行处理
    success_count = process_commit_semgrep_rules(
        commit_dir_path=COMMIT_DIR_PATH,
        generation_count=GENERATION_COUNT,
        use_langsmith=USE_LANGSMITH,
        reuse_existing=REUSE_EXISTING,
        max_round=MAX_ROUND
    )
    
    print_info(f"\n{'='*60}")
    print_info(f"MAIN FUNCTION COMPLETED")
    print_info(f"{'='*60}")
    print_info(f"Successfully generated/reused: {success_count}/{GENERATION_COUNT} rules")

if __name__ == "__main__":
    main()