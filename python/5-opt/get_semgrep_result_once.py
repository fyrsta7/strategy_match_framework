import os
import json
import subprocess
from import_configs import global_config, opt_config

def print_info(message):
    """关键信息输出 - 始终显示"""
    print(message)

def print_verbose(message):
    """详细信息输出 - 仅在VERBOSE模式下显示"""
    if opt_config.VERBOSE:
        print(message)

def run_semgrep_on_file(yaml_file_path, target_file_path, json_output_path, verbose=None):
    """
    使用指定的Semgrep YAML规则文件对目标代码文件进行扫描
    
    Args:
        yaml_file_path: Semgrep规则的YAML文件路径
        target_file_path: 待扫描的代码文件路径
        json_output_path: 结果JSON文件的保存路径
        verbose: 可选，覆盖全局VERBOSE设置
    
    Returns:
        tuple: (success: bool, error_message: str|None)
    """
    # 临时保存原始verbose设置
    original_verbose = opt_config.VERBOSE
    if verbose is not None:
        opt_config.VERBOSE = verbose
    
    print_verbose("Running Semgrep scan...")
    print_verbose(f"YAML rule file: {yaml_file_path}")
    print_verbose(f"Target file: {target_file_path}")
    print_verbose(f"Output JSON: {json_output_path}")
    
    # 输入验证
    if not os.path.exists(yaml_file_path):
        error_msg = f"YAML rule file not found: {yaml_file_path}"
        print_info(f"Error: {error_msg}")
        opt_config.VERBOSE = original_verbose
        return False, error_msg
    
    if not os.path.exists(target_file_path):
        error_msg = f"Target file not found: {target_file_path}"
        print_info(f"Error: {error_msg}")
        opt_config.VERBOSE = original_verbose
        return False, error_msg
    
    # 确保输出目录存在
    output_dir = os.path.dirname(json_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 构建Semgrep命令
    cmd = [
        "semgrep",
        f"--config={yaml_file_path}",
        target_file_path,
        "--json"
    ]
    
    try:
        print_verbose("Executing Semgrep command...")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=opt_config.TIMEOUT
        )
        
        print_verbose(f"Semgrep execution completed with return code: {result.returncode}")
        
        # 处理输出结果
        if result.stdout.strip():
            try:
                # 解析JSON输出
                json_result = json.loads(result.stdout)
                print_verbose("Successfully parsed JSON output")
                
                # 保存JSON结果
                with open(json_output_path, 'w') as f:
                    json.dump(json_result, f, indent=opt_config.DEFAULT_OUTPUT_INDENT)
                
                # 显示扫描统计
                results_count = len(json_result.get('results', []))
                errors_count = len(json_result.get('errors', []))
                print_info(f"Scan completed: {results_count} findings, {errors_count} errors")
                
                opt_config.VERBOSE = original_verbose
                return True, None
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON output: {e}"
                print_info(f"Error: {error_msg}")
                print_verbose(f"Raw stdout: {result.stdout}")
                print_verbose(f"Raw stderr: {result.stderr}")
                
                # 创建包含错误信息的JSON结果
                error_result = {
                    "version": "unknown",
                    "results": [],
                    "errors": [
                        {
                            "code": result.returncode,
                            "level": "error",
                            "type": "JSON parse error",
                            "message": f"Failed to parse JSON: {str(e)}",
                            "raw_stdout": result.stdout,
                            "raw_stderr": result.stderr
                        }
                    ],
                    "paths": {"scanned": [target_file_path]},
                    "skipped_rules": []
                }
                
                with open(json_output_path, 'w') as f:
                    json.dump(error_result, f, indent=opt_config.DEFAULT_OUTPUT_INDENT)
                
                opt_config.VERBOSE = original_verbose
                return False, error_msg
        
        else:
            # 无输出情况，创建默认结构
            print_verbose("No stdout output, creating default result structure")
            
            default_result = {
                "version": "unknown",
                "results": [],
                "errors": [],
                "paths": {"scanned": [target_file_path]},
                "skipped_rules": []
            }
            
            # 如果有stderr，添加到错误信息中
            if result.stderr.strip():
                default_result["errors"].append({
                    "code": result.returncode,
                    "level": "error",
                    "type": "Execution error",
                    "message": result.stderr
                })
                print_verbose(f"Stderr output: {result.stderr}")
            
            with open(json_output_path, 'w') as f:
                json.dump(default_result, f, indent=opt_config.DEFAULT_OUTPUT_INDENT)
            
            print_info("Scan completed: 0 findings")
            opt_config.VERBOSE = original_verbose
            return True, None
            
    except subprocess.TimeoutExpired:
        error_msg = f"Semgrep execution timed out ({opt_config.TIMEOUT} seconds)"
        print_info(f"Error: {error_msg}")
        
        # 创建超时错误的JSON结果
        timeout_result = {
            "version": "unknown",
            "results": [],
            "errors": [
                {
                    "code": -1,
                    "level": "error",
                    "type": "Timeout error",
                    "message": f"Semgrep execution timed out after {opt_config.TIMEOUT} seconds"
                }
            ],
            "paths": {"scanned": [target_file_path]},
            "skipped_rules": []
        }
        
        with open(json_output_path, 'w') as f:
            json.dump(timeout_result, f, indent=opt_config.DEFAULT_OUTPUT_INDENT)
        
        opt_config.VERBOSE = original_verbose
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Error running Semgrep: {str(e)}"
        print_info(f"Error: {error_msg}")
        
        # 创建通用错误的JSON结果
        general_error_result = {
            "version": "unknown",
            "results": [],
            "errors": [
                {
                    "code": -1,
                    "level": "error",
                    "type": "General execution error",
                    "message": str(e)
                }
            ],
            "paths": {"scanned": [target_file_path]},
            "skipped_rules": []
        }
        
        with open(json_output_path, 'w') as f:
            json.dump(general_error_result, f, indent=opt_config.DEFAULT_OUTPUT_INDENT)
        
        opt_config.VERBOSE = original_verbose
        return False, error_msg

def main():
    """
    主函数：使用默认参数进行测试
    """
    # 默认测试参数
    COMMIT1_NUM = 2  # 规则来源的commit
    COMMIT2_NUM = 2  # 测试目标的commit
    YAML_FILE_NAME = "2.yaml"  # 假设使用第一个生成的规则
    BEFORE_FUNC_FILE_NAME = "before_func.cc"
    
    # 构建文件路径
    yaml_file_path = f"{opt_config.semgrep_path}/yaml/{COMMIT1_NUM}/{YAML_FILE_NAME}"
    target_file_path = f"{opt_config.semgrep_path}/commit_info/{COMMIT2_NUM}/{BEFORE_FUNC_FILE_NAME}"
    json_output_path = f"{opt_config.semgrep_path}/cross_test/commit{COMMIT1_NUM}_rule_on_commit{COMMIT2_NUM}.json"
    
    print_info("=== Running Semgrep Cross-Test ===")
    print_verbose(f"Using rule from commit {COMMIT1_NUM} to scan code from commit {COMMIT2_NUM}")
    print_verbose(f"YAML rule: {yaml_file_path}")
    print_verbose(f"Target file: {target_file_path}")
    print_verbose(f"Output JSON: {json_output_path}")
    
    # 验证输入文件
    if not os.path.exists(yaml_file_path):
        print_info(f"Error: YAML rule file not found: {yaml_file_path}")
        return
    
    if not os.path.exists(target_file_path):
        print_info(f"Error: Target file not found: {target_file_path}")
        return
    
    # 执行Semgrep扫描
    success, error_message = run_semgrep_on_file(
        yaml_file_path=yaml_file_path,
        target_file_path=target_file_path,
        json_output_path=json_output_path
    )
    
    # 显示结果
    print_info("\n=== Execution Result ===")
    if success:
        print_info("SUCCESS: Semgrep scan completed successfully")
        print_verbose(f"Results saved to: {json_output_path}")
        
        # 尝试显示简要结果统计
        try:
            with open(json_output_path, 'r') as f:
                result_data = json.load(f)
            
            results_count = len(result_data.get('results', []))
            errors_count = len(result_data.get('errors', []))
            print_info(f"Summary: {results_count} findings detected, {errors_count} errors reported")
            
            # 如果有发现结果，显示简要信息
            if results_count > 0:
                print_verbose("Findings preview:")
                for i, result in enumerate(result_data.get('results', [])[:3]):  # 显示前3个结果
                    rule_id = result.get('check_id', 'unknown')
                    message = result.get('extra', {}).get('message', 'No message')
                    line = result.get('start', {}).get('line', 'unknown')
                    print_verbose(f"  {i+1}. Rule: {rule_id}, Line: {line}")
                    print_verbose(f"     Message: {message}")
                
                if results_count > 3:
                    print_verbose(f"  ... and {results_count - 3} more findings")
        
        except Exception as e:
            print_verbose(f"Note: Could not parse result summary: {e}")
            
    else:
        print_info("FAILED: Semgrep scan encountered errors")
        print_info(f"Error: {error_message}")
        print_verbose(f"Error details saved to: {json_output_path}")

if __name__ == "__main__":
    main()