import os
import json
import time
import math
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from import_configs import global_config, opt_config
from get_semgrep_result_commit import process_all_semgrep_rules

def print_info(message):
    """关键信息输出 - 始终显示"""
    print(message)

def print_verbose(message):
    """详细信息输出 - 仅在VERBOSE模式下显示"""
    if opt_config.VERBOSE:
        print(message)

# ============================================================================
# 数据结构定义
# ============================================================================
@dataclass
class CodeRange:
    start_line: int
    end_line: int

@dataclass
class MatchResult:
    rule_file: str
    rule_number: int
    check_id: str
    start_line: int
    end_line: int
    message: str
    hits_target: bool
    coverage_details: dict

@dataclass
class AnalysisResult:
    summary: dict
    matching_rules: List[dict]
    coverage_config: dict

# ============================================================================
# 辅助函数
# ============================================================================
def generate_temp_results_dir(rules_commit_dir, target_commit_file):
    """
    生成临时结果目录路径，确保唯一性
    
    Args:
        rules_commit_dir: 规则目录路径
        target_commit_file: 目标文件路径
    
    Returns:
        str: 临时目录路径
    """
    # 使用完整路径的哈希确保唯一性
    rules_path_hash = hashlib.md5(os.path.abspath(rules_commit_dir).encode()).hexdigest()[:12]
    target_path_hash = hashlib.md5(os.path.abspath(target_commit_file).encode()).hexdigest()[:12]
    
    temp_dir_name = f"cross_analysis_rules_{rules_path_hash}_target_{target_path_hash}"
    temp_dir_path = os.path.join(opt_config.semgrep_path, "temp", temp_dir_name)
    
    return temp_dir_path

def run_semgrep_batch_if_needed(rules_commit_dir, target_commit_file, temp_results_dir):
    """
    按需运行批量semgrep分析
    
    Args:
        rules_commit_dir: 规则目录路径
        target_commit_file: 目标文件路径
        temp_results_dir: 临时结果目录
    
    Returns:
        tuple: (success, error_message)
    """
    # 检查是否已有结果
    if os.path.exists(temp_results_dir) and os.listdir(temp_results_dir):
        print_verbose(f"Using existing results in: {temp_results_dir}")
        return True, None
    
    print_info("Running batch Semgrep analysis...")
    print_verbose(f"Rules directory: {rules_commit_dir}")
    print_verbose(f"Target file: {target_commit_file}")
    print_verbose(f"Output directory: {temp_results_dir}")
    
    # 调用批量处理函数
    total_count, success_count, failed_count, skipped_count, failed_files = process_all_semgrep_rules(
        yaml_dir_path=rules_commit_dir,
        target_file_path=target_commit_file,
        json_output_dir=temp_results_dir,
        skip_existing=opt_config.SEMGREP_COMMIT_SKIP_EXISTING
    )
    
    if total_count == 0:
        return False, "No rules found to process"
    
    if success_count == 0:
        return False, f"All {total_count} rules failed to execute"
    
    print_info(f"Batch processing completed: {success_count}/{total_count} rules succeeded")
    return True, None

def cleanup_temp_results(temp_results_dir):
    """
    清理临时结果文件
    
    Args:
        temp_results_dir: 临时结果目录
    """
    if not opt_config.ANALYZE_CLEANUP_TEMP_FILES:
        return
    
    try:
        import shutil
        if os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)
            print_verbose(f"Cleaned up temporary directory: {temp_results_dir}")
    except Exception as e:
        print_verbose(f"Failed to cleanup temporary directory: {e}")

# ============================================================================
# 核心分析函数
# ============================================================================
def calculate_coverage_requirements(target_range, coverage_config):
    """
    计算覆盖要求
    
    Args:
        target_range: 目标代码范围
        coverage_config: 覆盖配置
    
    Returns:
        dict: 计算后的覆盖要求
    """
    target_length = target_range.end_line - target_range.start_line + 1
    
    min_overlap_lines = coverage_config.get("min_overlap_lines", opt_config.ANALYZE_MIN_OVERLAP_LINES)
    min_coverage_ratio = coverage_config.get("min_coverage_ratio", opt_config.ANALYZE_MIN_COVERAGE_RATIO)
    
    # 计算基于比例的最小重叠行数
    ratio_required_lines = math.ceil(target_length * min_coverage_ratio)
    
    # 取绝对值和比例值的最大值
    actual_min_overlap = max(min_overlap_lines, ratio_required_lines)
    
    max_overage_ratio = coverage_config.get("max_overage_ratio", opt_config.ANALYZE_MAX_OVERAGE_RATIO)
    max_allowed_length = target_length * max_overage_ratio
    
    return {
        "target_length": target_length,
        "min_overlap_lines": actual_min_overlap,
        "max_allowed_length": max_allowed_length,
        "allow_partial_coverage": coverage_config.get("allow_partial_coverage", opt_config.ANALYZE_ALLOW_PARTIAL_COVERAGE),
        "require_exact_boundaries": coverage_config.get("require_exact_boundaries", opt_config.ANALYZE_REQUIRE_EXACT_BOUNDARIES)
    }

def check_coverage_requirements(match_start, match_end, target_range, coverage_requirements):
    """
    检查匹配结果是否满足覆盖要求
    
    Args:
        match_start: 匹配起始行
        match_end: 匹配结束行  
        target_range: 目标范围
        coverage_requirements: 覆盖要求
    
    Returns:
        tuple: (是否满足要求, 详细信息)
    """
    target_start = target_range.start_line
    target_end = target_range.end_line
    
    # 如果要求精确边界匹配
    if coverage_requirements["require_exact_boundaries"]:
        exact_match = (match_start == target_start and match_end == target_end)
        return exact_match, {
            "exact_boundary_match": exact_match,
            "match_range": f"{match_start}-{match_end}",
            "target_range": f"{target_start}-{target_end}",
            "meets_requirements": exact_match
        }
    
    # 计算重叠范围
    overlap_start = max(match_start, target_start)
    overlap_end = min(match_end, target_end)
    
    if overlap_start > overlap_end:
        return False, {
            "overlap_lines": 0,
            "match_range": f"{match_start}-{match_end}",
            "target_range": f"{target_start}-{target_end}",
            "meets_requirements": False,
            "reason": "no_overlap"
        }
    
    overlap_lines = overlap_end - overlap_start + 1
    match_length = match_end - match_start + 1
    
    # 检查最小重叠要求
    meets_min_overlap = overlap_lines >= coverage_requirements["min_overlap_lines"]
    
    # 检查最大长度限制
    meets_max_length = match_length <= coverage_requirements["max_allowed_length"]
    
    # 检查完全覆盖要求
    if not coverage_requirements["allow_partial_coverage"]:
        full_coverage = (overlap_start <= target_start and overlap_end >= target_end)
        meets_coverage = full_coverage
    else:
        meets_coverage = True
    
    meets_all_requirements = meets_min_overlap and meets_max_length and meets_coverage
    
    details = {
        "overlap_lines": overlap_lines,
        "required_overlap_lines": coverage_requirements["min_overlap_lines"],
        "match_length": match_length,
        "max_allowed_length": coverage_requirements["max_allowed_length"],
        "overage_ratio": match_length / coverage_requirements["target_length"],
        "match_range": f"{match_start}-{match_end}",
        "target_range": f"{target_start}-{target_end}",
        "overlap_range": f"{overlap_start}-{overlap_end}",
        "meets_min_overlap": meets_min_overlap,
        "meets_max_length": meets_max_length,
        "meets_coverage": meets_coverage,
        "meets_requirements": meets_all_requirements
    }
    
    return meets_all_requirements, details

def scan_semgrep_results(semgrep_results_dir):
    """
    扫描Semgrep结果文件，排除analyze_result.json
    
    Args:
        semgrep_results_dir: 结果目录路径
    
    Returns:
        List[str]: JSON文件路径列表
    """
    json_files = []
    
    if not os.path.exists(semgrep_results_dir):
        return json_files
    
    for filename in os.listdir(semgrep_results_dir):
        if (filename.endswith('.json') and 
            not filename.endswith('.error') and 
            not filename.startswith('analyze_result')):
            json_path = os.path.join(semgrep_results_dir, filename)
            json_files.append(json_path)
    
    # 按文件名数字排序
    try:
        json_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    except ValueError:
        json_files.sort()
    
    return json_files

def parse_semgrep_json(json_file_path):
    """
    解析单个Semgrep JSON文件
    
    Args:
        json_file_path: JSON文件路径
    
    Returns:
        tuple: (是否成功, 结果列表, 错误信息)
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if not isinstance(json_data, dict):
            return False, [], "Invalid JSON structure"
        
        results = json_data.get("results", [])
        parsed_results = []
        
        for result in results:
            try:
                parsed_result = {
                    "check_id": result.get("check_id", "unknown"),
                    "start_line": result["start"]["line"],
                    "end_line": result["end"]["line"],
                    "message": result.get("extra", {}).get("message", "")
                }
                parsed_results.append(parsed_result)
            except KeyError as e:
                continue  # 跳过格式不正确的结果
        
        return True, parsed_results, None
        
    except json.JSONDecodeError as e:
        return False, [], f"JSON decode error: {str(e)}"
    except Exception as e:
        return False, [], f"Unexpected error: {str(e)}"

def generate_analysis_result(matching_rules, total_rules_analyzed, target_range, coverage_config, 
                           rules_commit_dir, target_commit_file):
    """
    生成分析结果
    
    Args:
        matching_rules: 匹配的规则列表
        total_rules_analyzed: 分析的总规则数
        target_range: 目标范围
        coverage_config: 覆盖配置
        rules_commit_dir: 规则目录路径
        target_commit_file: 目标文件路径
    
    Returns:
        AnalysisResult: 分析结果
    """
    summary = {
        "has_matching_rules": len(matching_rules) > 0,
        "total_rules_analyzed": total_rules_analyzed,
        "matching_rules_count": len(matching_rules),
        "target_range": f"{target_range.start_line}-{target_range.end_line}",
        "rules_source": rules_commit_dir,
        "target_file": target_commit_file,
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return AnalysisResult(
        summary=summary,
        matching_rules=matching_rules,
        coverage_config=coverage_config
    )

def save_analysis_result(analysis_result, output_file_path):
    """
    保存分析结果到JSON文件
    
    Args:
        analysis_result: 分析结果
        output_file_path: 输出文件路径
    
    Returns:
        bool: 是否保存成功
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        result_data = {
            "summary": analysis_result.summary,
            "matching_rules": analysis_result.matching_rules,
            "coverage_config": analysis_result.coverage_config
        }
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=opt_config.DEFAULT_OUTPUT_INDENT, ensure_ascii=False)
        
        return True
    except Exception as e:
        print_info(f"Failed to save analysis result: {e}")
        return False

def analyze_cross_commit_semgrep_coverage(rules_commit_dir, target_commit_file, 
                                        target_start_line, target_end_line, 
                                        coverage_config=None) -> Tuple[bool, str]:
    """
    分析跨commit的Semgrep覆盖情况
    
    Args:
        rules_commit_dir: Semgrep规则文件夹路径（commit1）
        target_commit_file: 目标代码文件路径（commit2）
        target_start_line: 目标代码片段起始行号
        target_end_line: 目标代码片段结束行号
        coverage_config: 覆盖要求配置字典
    
    Returns:
        tuple: (是否存在符合要求的规则: bool, 临时结果目录路径: str)
    """
    # 设置默认配置
    if coverage_config is None:
        coverage_config = {
            "min_overlap_lines": opt_config.ANALYZE_MIN_OVERLAP_LINES,
            "min_coverage_ratio": opt_config.ANALYZE_MIN_COVERAGE_RATIO,
            "max_overage_ratio": opt_config.ANALYZE_MAX_OVERAGE_RATIO,
            "allow_partial_coverage": opt_config.ANALYZE_ALLOW_PARTIAL_COVERAGE,
            "require_exact_boundaries": opt_config.ANALYZE_REQUIRE_EXACT_BOUNDARIES
        }
    
    print_info("=== Cross-Commit Semgrep Coverage Analysis ===")
    print_verbose(f"Rules directory: {rules_commit_dir}")
    print_verbose(f"Target file: {target_commit_file}")
    print_verbose(f"Target range: {target_start_line}-{target_end_line}")
    print_verbose(f"Coverage config: {coverage_config}")
    
    # 生成临时结果目录（无论后续是否成功都要返回这个路径）
    temp_results_dir = generate_temp_results_dir(rules_commit_dir, target_commit_file)
    print_verbose(f"Temporary results directory: {temp_results_dir}")
    
    # 输入验证
    if not os.path.exists(rules_commit_dir):
        print_info(f"Error: Rules directory not found: {rules_commit_dir}")
        return False, temp_results_dir
    
    if not os.path.exists(target_commit_file):
        print_info(f"Error: Target file not found: {target_commit_file}")
        return False, temp_results_dir
    
    target_range = CodeRange(start_line=target_start_line, end_line=target_end_line)
    
    # 计算覆盖要求
    coverage_requirements = calculate_coverage_requirements(target_range, coverage_config)
    print_verbose(f"Required overlap lines: {coverage_requirements['min_overlap_lines']}")
    
    # 运行批量Semgrep分析
    batch_success, batch_error = run_semgrep_batch_if_needed(
        rules_commit_dir, target_commit_file, temp_results_dir
    )
    
    if not batch_success:
        print_info(f"Error: Batch Semgrep analysis failed: {batch_error}")
        # 仍然保存空结果到文件
        analysis_result = generate_analysis_result(
            [], 0, target_range, coverage_config, rules_commit_dir, target_commit_file
        )
        output_file = os.path.join(temp_results_dir, "analyze_result.json")
        save_analysis_result(analysis_result, output_file)
        return False, temp_results_dir
    
    # 扫描JSON文件
    json_files = scan_semgrep_results(temp_results_dir)
    
    if not json_files:
        print_info(f"No JSON result files found in {temp_results_dir}")
        # 保存空结果
        analysis_result = generate_analysis_result(
            [], 0, target_range, coverage_config, rules_commit_dir, target_commit_file
        )
        output_file = os.path.join(temp_results_dir, "analyze_result.json")
        save_analysis_result(analysis_result, output_file)
        return False, temp_results_dir
    
    print_info(f"Analyzing {len(json_files)} semgrep result files...")
    
    # 分析每个文件
    matching_rules = []
    total_rules_analyzed = 0
    
    # 根据VERBOSE设置决定是否显示进度条
    iterator = tqdm(json_files, desc="Processing rules") if opt_config.VERBOSE else json_files
    
    for json_file in iterator:
        rule_number = None
        try:
            filename = os.path.basename(json_file)
            rule_number = int(filename.split('.')[0])
        except ValueError:
            rule_number = total_rules_analyzed + 1
        
        total_rules_analyzed += 1
        
        # 解析JSON文件
        success, results, error_msg = parse_semgrep_json(json_file)
        
        if not success:
            print_verbose(f"Failed to parse {json_file}: {error_msg}")
            continue
        
        # 检查每个匹配结果
        rule_matches = []
        rule_hits_target = False
        
        for result in results:
            meets_requirements, coverage_details = check_coverage_requirements(
                result["start_line"], result["end_line"], 
                target_range, coverage_requirements
            )
            
            if meets_requirements:
                rule_hits_target = True
                match_info = {
                    "start_line": result["start_line"],
                    "end_line": result["end_line"],
                    "check_id": result["check_id"],
                    "coverage_details": coverage_details
                }
                rule_matches.append(match_info)
        
        # 如果该规则有符合要求的匹配
        if rule_hits_target:
            matching_rule = {
                "rule_file": os.path.basename(json_file),
                "rule_number": rule_number,
                "match_ranges": [f"{m['start_line']}-{m['end_line']}" for m in rule_matches],
                "coverage_details": {
                    "total_matches": len(rule_matches),
                    "meets_all_requirements": True,
                    "match_details": rule_matches
                }
            }
            matching_rules.append(matching_rule)
    
    # 生成分析结果
    analysis_result = generate_analysis_result(
        matching_rules, total_rules_analyzed, target_range, coverage_config,
        rules_commit_dir, target_commit_file
    )
    
    # 保存结果到文件
    output_file = os.path.join(temp_results_dir, "analyze_result.json")
    save_success = save_analysis_result(analysis_result, output_file)
    
    # 输出结果摘要
    has_matching_rules = len(matching_rules) > 0
    print_info(f"\nAnalysis completed:")
    print_info(f"  Total rules analyzed: {total_rules_analyzed}")
    print_info(f"  Matching rules: {len(matching_rules)}")
    print_info(f"  Result: {'✅ FOUND' if has_matching_rules else '❌ NOT FOUND'}")
    
    if save_success:
        print_info(f"  Detailed results saved to: {output_file}")
    
    print_info(f"  Temporary directory: {temp_results_dir}")
    
    # 注意：不再自动清理临时文件，因为调用者可能需要使用返回的路径
    # cleanup_temp_results(temp_results_dir)
    
    return has_matching_rules, temp_results_dir

# ============================================================================
# 测试功能
# ============================================================================
def main():
    """
    测试主函数
    """
    # 测试配置
    RULES_COMMIT_NUM = 1    # 规则来源commit
    TARGET_COMMIT_NUM = 1   # 目标代码commit
    TARGET_START_LINE = 204
    TARGET_END_LINE = 209
    
    # 构建路径
    test_rules_dir = f"{opt_config.semgrep_path}/yaml/{RULES_COMMIT_NUM}"
    test_target_file = f"{opt_config.semgrep_path}/commit_info/{TARGET_COMMIT_NUM}/before.cc"
    
    test_coverage_config = {
        "min_overlap_lines": opt_config.ANALYZE_MIN_OVERLAP_LINES,
        "min_coverage_ratio": opt_config.ANALYZE_MIN_COVERAGE_RATIO,
        "max_overage_ratio": opt_config.ANALYZE_MAX_OVERAGE_RATIO,
        "allow_partial_coverage": opt_config.ANALYZE_ALLOW_PARTIAL_COVERAGE,
        "require_exact_boundaries": opt_config.ANALYZE_REQUIRE_EXACT_BOUNDARIES
    }
    
    print_info("=" * 60)
    print_info("CROSS-COMMIT SEMGREP COVERAGE ANALYZER - TEST")
    print_info("=" * 60)
    print_verbose(f"Rules directory: {test_rules_dir}")
    print_verbose(f"Target file: {test_target_file}")
    print_verbose(f"Target range: {TARGET_START_LINE}-{TARGET_END_LINE}")
    print_verbose(f"Coverage config: {test_coverage_config}")
    print_info("")
    
    # 执行分析
    has_matching_rules, temp_dir = analyze_cross_commit_semgrep_coverage(
        rules_commit_dir=test_rules_dir,
        target_commit_file=test_target_file,
        target_start_line=TARGET_START_LINE,
        target_end_line=TARGET_END_LINE,
        coverage_config=test_coverage_config
    )
    
    print_info(f"\n🎯 Final Result:")
    print_info(f"  Has matching rules: {has_matching_rules}")
    print_info(f"  Temporary directory: {temp_dir}")

if __name__ == "__main__":
    main()