import os
import json
import glob
from import_configs import global_config, opt_config
from get_semgrep_result_once import run_semgrep_on_file

def print_info(message):
    """关键信息输出 - 始终显示"""
    print(message)

def print_verbose(message):
    """详细信息输出 - 仅在VERBOSE模式下显示"""
    if opt_config.VERBOSE:
        print(message)

def find_yaml_files(yaml_dir_path):
    """
    扫描目录中的所有YAML文件
    
    Args:
        yaml_dir_path: YAML规则文件所在目录
    
    Returns:
        List[str]: YAML文件路径列表，按文件名排序
    """
    yaml_files = glob.glob(os.path.join(yaml_dir_path, "*.yaml"))
    yaml_files.extend(glob.glob(os.path.join(yaml_dir_path, "*.yml")))
    return sorted(yaml_files)

def should_skip_file(json_file_path, skip_existing):
    """
    检查是否应该跳过当前文件的处理
    
    Args:
        json_file_path: 输出JSON文件路径
        skip_existing: 是否跳过已存在文件的标志
    
    Returns:
        bool: 是否应该跳过此文件
    """
    if not skip_existing:
        return False
    
    if not os.path.exists(json_file_path):
        return False
    
    # 检查文件是否为空
    if os.path.getsize(json_file_path) == 0:
        return False
    
    # 检查是否为有效JSON
    try:
        with open(json_file_path, 'r') as f:
            json.load(f)
        return True
    except:
        # 如果JSON无效，重新处理
        return False

def process_all_semgrep_rules(yaml_dir_path, target_file_path, json_output_dir, skip_existing=None):
    """
    批量执行Semgrep规则扫描
    
    Args:
        yaml_dir_path: Semgrep规则文件夹路径
        target_file_path: 待扫描的代码文件路径
        json_output_dir: JSON结果文件夹路径
        skip_existing: 是否跳过已存在的非空JSON文件，None时使用config默认值
    
    Returns:
        tuple: (total_count, success_count, failed_count, skipped_count, failed_files)
    """
    if skip_existing is None:
        skip_existing = opt_config.SEMGREP_COMMIT_SKIP_EXISTING
    
    print_info(f"=== Processing All Semgrep Rules ===")
    print_verbose(f"YAML directory: {yaml_dir_path}")
    print_verbose(f"Target file: {target_file_path}")
    print_verbose(f"Output directory: {json_output_dir}")
    print_verbose(f"Skip existing: {skip_existing}")
    print_verbose(f"Timeout setting: {opt_config.TIMEOUT} seconds")
    
    # 输入验证
    if not os.path.exists(yaml_dir_path):
        print_info(f"Error: YAML directory not found: {yaml_dir_path}")
        return 0, 0, 0, 0, []
    
    if not os.path.exists(target_file_path):
        print_info(f"Error: Target file not found: {target_file_path}")
        return 0, 0, 0, 0, []
    
    # 确保输出目录存在
    os.makedirs(json_output_dir, exist_ok=True)
    
    # 找到所有YAML文件
    yaml_files = find_yaml_files(yaml_dir_path)
    total_count = len(yaml_files)
    
    if total_count == 0:
        print_info("No YAML files found in directory")
        return 0, 0, 0, 0, []
    
    print_info(f"Found {total_count} YAML files in directory")
    if opt_config.VERBOSE:
        print()
    
    # 统计变量
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_files = []
    
    # 处理每个YAML文件
    for i, yaml_file_path in enumerate(yaml_files, 1):
        yaml_filename = os.path.basename(yaml_file_path)
        yaml_name = os.path.splitext(yaml_filename)[0]
        json_filename = f"{yaml_name}.json"
        json_file_path = os.path.join(json_output_dir, json_filename)
        
        # 进度显示控制
        if opt_config.VERBOSE:
            print_verbose(f"Processing [{i}/{total_count}]: {yaml_filename} -> {json_filename}")
        elif i % opt_config.SEMGREP_COMMIT_PROGRESS_INTERVAL == 0 or i == total_count:
            print_info(f"Progress: [{i}/{total_count}] files processed")
        
        # 检查是否应该跳过
        if should_skip_file(json_file_path, skip_existing):
            print_verbose(f"  ⏩ Skipped (output file already exists and is non-empty)")
            skipped_count += 1
            continue
        
        # 执行Semgrep扫描，在批量处理时关闭详细输出
        success, error_message = run_semgrep_on_file(
            yaml_file_path=yaml_file_path,
            target_file_path=target_file_path,
            json_output_path=json_file_path,
            verbose=False
        )
        
        if success:
            print_verbose(f"  ✓ Completed successfully")
            success_count += 1
        else:
            print_verbose(f"  ✗ Failed: {error_message}")
            failed_count += 1
            failed_files.append(yaml_filename)
        
        if opt_config.VERBOSE:
            print_verbose("")
    
    # 显示最终统计
    print_info("\n=== Execution Summary ===")
    print_info(f"Total files: {total_count}")
    print_info(f"Success: {success_count}")
    print_info(f"Failed: {failed_count}")
    print_info(f"Skipped: {skipped_count}")
    
    if failed_files:
        print_info(f"Failed files: {failed_files}")
    
    return total_count, success_count, failed_count, skipped_count, failed_files

def main():
    """
    主函数：使用默认参数进行测试
    """
    # 默认测试参数
    COMMIT1_NUM = 4  # 规则来源的commit
    COMMIT2_NUM = 3  # 测试目标的commit
    SKIP_EXISTING = opt_config.SEMGREP_COMMIT_SKIP_EXISTING  # 使用config默认值
    
    # 构建文件路径，使用config中的路径
    yaml_dir_path = f"{opt_config.semgrep_path}/yaml/{COMMIT1_NUM}"
    target_file_path = f"{opt_config.semgrep_path}/commit_info/{COMMIT2_NUM}/before.cc"
    json_output_dir = f"{opt_config.semgrep_path}/batch_results/commit{COMMIT1_NUM}_rules_on_commit{COMMIT2_NUM}"
    
    print_info("=== Running Batch Semgrep Processing ===")
    print_verbose(f"Processing rules from commit {COMMIT1_NUM} on code from commit {COMMIT2_NUM}")
    print_verbose(f"YAML rules directory: {yaml_dir_path}")
    print_verbose(f"Target file: {target_file_path}")
    print_verbose(f"Output directory: {json_output_dir}")
    print_verbose(f"Skip existing files: {SKIP_EXISTING}")
    print_verbose("")
    
    # 验证输入路径
    if not os.path.exists(yaml_dir_path):
        print_info(f"Error: YAML rules directory not found: {yaml_dir_path}")
        return
    
    if not os.path.exists(target_file_path):
        print_info(f"Error: Target file not found: {target_file_path}")
        return
    
    # 执行批量处理
    total_count, success_count, failed_count, skipped_count, failed_files = process_all_semgrep_rules(
        yaml_dir_path=yaml_dir_path,
        target_file_path=target_file_path,
        json_output_dir=json_output_dir,
        skip_existing=SKIP_EXISTING
    )
    
    # 显示最终结果
    print_info(f"\n=== Final Result ===")
    if total_count == 0:
        print_info("No files to process")
    elif failed_count == 0:
        print_info(f"ALL SUCCESSFUL: {success_count} rules processed successfully, {skipped_count} skipped")
    else:
        print_info(f"PARTIALLY SUCCESSFUL: {success_count} success, {failed_count} failed, {skipped_count} skipped")
        print_info(f"Check individual error messages above for details")

if __name__ == "__main__":
    main()