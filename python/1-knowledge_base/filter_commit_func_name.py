#!/usr/bin/env python3
"""
筛选代码库中的commit，过滤掉修改单元测试相关函数的commit
"""
import json
import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局变量，定义输入和输出的JSON文件名
INPUT_JSON_FILENAME = "func_name.json"           # 输入文件名
OUTPUT_JSON_FILENAME = "func_name_result.json"   # 输出文件名

# 全局变量，控制是否跳过已经处理过的代码库
SKIP_EXISTING_RESULTS = False

# 并行处理的线程数
MAX_WORKERS = 64

# 全局变量，knowledge_base路径
KNOWLEDGE_BASE_PATH = os.path.join(config.root_path, "knowledge_base_all")

# 单元测试相关的函数名称列表（不影响性能测试结果的函数）
UNIT_TEST_FUNCTION_PATTERNS = [
    # Google Test框架
    "TEST",
    "TEST_F", 
    "TEST_P",
    "TYPED_TEST",
    "TYPED_TEST_P",
    "INSTANTIATE_TEST_CASE_P",
    "INSTANTIATE_TEST_SUITE_P",
    "INSTANTIATE_TYPED_TEST_CASE_P",
    "INSTANTIATE_TYPED_TEST_SUITE_P"
]

# UNIT_TEST_FUNCTION_PATTERNS = [
#     # Google Test框架
#     "TEST",
#     "TEST_F", 
#     "TEST_P",
#     "TYPED_TEST",
#     "TYPED_TEST_P",
#     "INSTANTIATE_TEST_CASE_P",
#     "INSTANTIATE_TEST_SUITE_P",
#     "INSTANTIATE_TYPED_TEST_CASE_P",
#     "INSTANTIATE_TYPED_TEST_SUITE_P",
    
#     # 其他测试框架常见模式
#     "test_",           # Python unittest风格
#     "Test",            # CamelCase测试函数
#     "testCase",        # camelCase测试函数
#     "TestCase",        # 测试用例类
#     "setUp",           # 测试设置
#     "tearDown",        # 测试清理
#     "setUpClass",      # 类级别设置
#     "tearDownClass",   # 类级别清理
#     "beforeEach",      # 每个测试前执行
#     "afterEach",       # 每个测试后执行
#     "beforeAll",       # 所有测试前执行
#     "afterAll",        # 所有测试后执行
    
#     # Mock和Stub相关
#     "Mock",
#     "Stub",
#     "Fake",
#     "mock_",
#     "stub_",
#     "fake_",
    
#     # 断言和验证函数
#     "Assert",
#     "Verify",
#     "Check",
#     "Expect",
#     "Should",
#     "assert_",
#     "verify_",
#     "check_",
#     "expect_",
    
#     # 测试辅助函数
#     "Helper",
#     "Utility",
#     "TestHelper",
#     "TestUtility",
#     "helper_",
#     "utility_",
#     "test_helper",
#     "test_util",
    
#     # 数据生成和fixture
#     "fixture",
#     "Fixture",
#     "createTest",
#     "generateTest",
#     "buildTest",
#     "makeTest",
    
#     # 常见的测试文件中的辅助函数
#     "setup",
#     "cleanup",
#     "initialize",
#     "finalize",
#     "prepare",
#     "reset",
    
#     # Catch2框架
#     "SCENARIO",
#     "GIVEN",
#     "WHEN",
#     "THEN",
#     "AND_WHEN",
#     "AND_THEN",
    
#     # Jest/Mocha风格 (JavaScript测试框架，但可能在C++中模仿)
#     "describe",
#     "it",
#     "beforeEach",
#     "afterEach",
    
#     # 其他常见测试相关模式
#     "validate",
#     "Validate",
#     "TestData",
#     "SampleData",
#     "DummyData",
#     "MockData",
# ]

def is_unit_test_function(func_name):
    """
    判断函数名是否为单元测试相关函数
    
    Args:
        func_name (str): 函数名称
        
    Returns:
        bool: 如果是单元测试相关函数返回True，否则返回False
    """
    if not func_name:
        return False
    
    # 如果没有配置任何模式，则不过滤任何函数
    if not UNIT_TEST_FUNCTION_PATTERNS:
        return False
    
    # 转换为字符串并去除空白字符
    func_name = str(func_name).strip()
    
    # 检查是否包含测试相关的模式
    for pattern in UNIT_TEST_FUNCTION_PATTERNS:
        if pattern.lower() in func_name.lower():
            return True
    
    return False

def filter_commits_by_function(input_file, output_file, repo_name):
    """
    从输入文件读取commit信息，过滤掉修改单元测试相关函数的commit
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        repo_name (str): 仓库名称
        
    Returns:
        tuple: (结果消息, 输入commit数量, 输出commit数量, 过滤的commit数量)
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            return f"[FuncFilter] 错误：输入文件 '{input_file}' 不存在。", 0, 0, 0
        
        # 读取输入文件中的commit信息
        with open(input_file, "r", encoding="utf-8") as file:
            all_commits = json.load(file)
        
        if not isinstance(all_commits, list):
            return f"[FuncFilter] 错误：输入文件格式错误，应为数组。", 0, 0, 0
        
        input_commit_count = len(all_commits)
        
        # 筛选非单元测试相关的commit
        filtered_commits = []
        filtered_out_commits = []
        
        for commit in all_commits:
            modified_func = commit.get("modified_func", [])
            
            # 检查所有修改的函数是否都不是单元测试相关
            is_test_related = False
            
            if isinstance(modified_func, list):
                for func_name in modified_func:
                    if is_unit_test_function(func_name):
                        is_test_related = True
                        break
            elif isinstance(modified_func, str):
                if is_unit_test_function(modified_func):
                    is_test_related = True
            
            if not is_test_related:
                # 保留非测试相关的commit
                filtered_commits.append(commit)
            else:
                # 记录被过滤的commit（用于调试）
                filtered_out_commits.append({
                    'hash': commit.get('hash', 'unknown'),
                    'modified_func': modified_func
                })
        
        output_commit_count = len(filtered_commits)
        filtered_count = input_commit_count - output_commit_count
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 将筛选结果写入输出文件
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(filtered_commits, file, indent=4, ensure_ascii=False)
        
        # 可选：保存被过滤的commit信息用于调试
        debug_file = output_file.replace('.json', '_filtered_out.json')
        with open(debug_file, "w", encoding="utf-8") as file:
            json.dump(filtered_out_commits, file, indent=4, ensure_ascii=False)
        
        return (f"[FuncFilter] 代码库 {repo_name}：从 {input_commit_count} 个commit中过滤掉 {filtered_count} 个测试相关commit，"
                f"保留 {output_commit_count} 个commit，结果已保存到 {output_file}",
                input_commit_count, output_commit_count, filtered_count)
        
    except Exception as e:
        return f"[FuncFilter] 处理 {repo_name} 时发生错误: {str(e)}", 0, 0, 0

def process_single_repository(repo, knowledge_base_path):
    """
    处理单个代码库的函数，用于并行执行
    
    Args:
        repo (str): 仓库名称
        knowledge_base_path (str): 知识库路径
        
    Returns:
        tuple: (结果消息, 输入数量, 输出数量, 过滤数量)
    """
    input_file = os.path.join(knowledge_base_path, INPUT_JSON_FILENAME)
    output_file = os.path.join(knowledge_base_path, OUTPUT_JSON_FILENAME)
    
    # 检查是否需要跳过当前代码库
    if SKIP_EXISTING_RESULTS and os.path.exists(output_file):
        return f"[FuncFilter] 文件 '{output_file}' 已存在，跳过代码库 {repo}。", 0, 0, 0
    
    return filter_commits_by_function(input_file, output_file, repo)

def process_function_filter_phase(repositories):
    """
    对所有代码库执行函数筛选阶段
    
    Args:
        repositories (list): 仓库列表
    """
    print("===== 函数筛选阶段 =====")
    print(f"输入文件: {INPUT_JSON_FILENAME}")
    print(f"输出文件: {OUTPUT_JSON_FILENAME}")
    print(f"单元测试函数模式数量: {len(UNIT_TEST_FUNCTION_PATTERNS)}")
    print("将过滤掉修改以下类型函数的commit:")
    for i, pattern in enumerate(UNIT_TEST_FUNCTION_PATTERNS[:10]):  # 只显示前10个
        print(f"  - {pattern}")
    if len(UNIT_TEST_FUNCTION_PATTERNS) > 10:
        print(f"  ... 还有 {len(UNIT_TEST_FUNCTION_PATTERNS) - 10} 个模式")
    
    # 准备任务列表
    tasks = []
    for repo in repositories:
        knowledge_base_path = os.path.join(KNOWLEDGE_BASE_PATH, repo)
        tasks.append((repo, knowledge_base_path))
    
    # 使用线程池并行处理
    results = []
    total_input_commits = 0
    total_output_commits = 0
    total_filtered_commits = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_repo = {
            executor.submit(process_single_repository, repo, knowledge_base_path): repo
            for repo, knowledge_base_path in tasks
        }
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_repo), total=len(tasks), desc="Function filtering"):
            repo = future_to_repo[future]
            try:
                result, input_count, output_count, filtered_count = future.result()
                results.append(result)
                total_input_commits += input_count
                total_output_commits += output_count
                total_filtered_commits += filtered_count
            except Exception as exc:
                error_msg = f"[FuncFilter] 代码库 {repo} 处理时发生异常: {exc}"
                results.append(error_msg)
                print(error_msg)
    
    # 打印所有结果
    for result in results:
        if result:  # 只打印非空结果
            print(result)
    
    # 打印总体统计信息
    print(f"\n===== 筛选统计 =====")
    print(f"输入文件总commit数: {total_input_commits}")
    print(f"输出文件总commit数: {total_output_commits}")
    print(f"过滤掉的commit数: {total_filtered_commits}")
    if total_input_commits > 0:
        keep_rate = (total_output_commits / total_input_commits) * 100
        filter_rate = (total_filtered_commits / total_input_commits) * 100
        print(f"保留率: {keep_rate:.2f}%")
        print(f"过滤率: {filter_rate:.2f}%")
    
    print(f"输入文件名: {INPUT_JSON_FILENAME}")
    print(f"输出文件名: {OUTPUT_JSON_FILENAME}")

def main():
    """
    主函数
    """
    # 排除不处理的仓库
    EXCLUDED_REPOSITORIES = []
    
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"Error: 目录 '{KNOWLEDGE_BASE_PATH}' 不存在。")
        sys.exit(1)
    
    repositories = [
        folder for folder in os.listdir(KNOWLEDGE_BASE_PATH)
        if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, folder)) and folder not in EXCLUDED_REPOSITORIES
    ]
    
    if not repositories:
        print("未找到任何代码仓库")
        return
    
    print(f"找到 {len(repositories)} 个代码库")
    
    # 执行函数筛选
    process_function_filter_phase(repositories)
    
    print("\n所有仓库处理完成！")

if __name__ == "__main__":
    main()