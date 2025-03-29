import os
import sys
import importlib.util
import subprocess
from pathlib import Path

def load_module(file_path, module_name):
    """动态加载Python模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_script(script_path):
    """执行Python脚本"""
    cmd = [sys.executable, script_path]
    print(f"执行脚本: {script_path}")
    subprocess.run(cmd, check=True)

def main():
    # 获取当前工作目录
    base_dir = os.getcwd()
    
    # 导入filter_commit_other模块
    filter_other_path = os.path.join(base_dir, "filter_commit_other.py")
    filter_other = load_module(filter_other_path, "filter_commit_other")
    
    # 按顺序执行所有步骤
    # print("步骤0: 获取所有commit")
    # filter_other.get_all_commits()

    # print("步骤1: 获取每个commit修改的文件数量")
    # run_script(os.path.join(base_dir, "get_commit_file_count.py"))
    
    # print("步骤2: 筛选出只修改一个文件的commit")
    # filter_other.filter_one_file_commits()
    
    # print("步骤3: 筛选C/C++语言的commit")
    # filter_other.filter_c_language_commits()
    
    # print("步骤4: 通过关键词筛选性能优化相关的commit")
    # run_script(os.path.join(base_dir, "filter_commit_keyword.py"))
    
    # print("步骤5: 获取commit文件修改前后的完整内容")
    # run_script(os.path.join(base_dir, "get_commit_file_changes.py"))
    
    # print("步骤6: 筛选出在modified_file目录存在的commit")
    # filter_other.get_has_file()
    
    # print("步骤7: 复制has_file.json到has_file_with_func.json")
    # filter_other.copy_has_file_to_func()
    
    # print("步骤8: 处理函数修改情况")
    # run_script(os.path.join(base_dir, "../1-diff/get_func_claude37.py"))
    
    # print("步骤9: 筛选出只修改一个函数的commit")
    # filter_other.filter_one_func_commits()
    
    # print("步骤10: 汇总并去重one_func.json")
    # filter_other.aggregate_and_deduplicate_one_func()
    
    # print("步骤11: 使用LLM筛选性能优化相关commit")
    # run_script(os.path.join(base_dir, "filter_commit_llm.py"))
    
    print("步骤12: 筛选最终的性能优化commit")
    filter_other.filter_optimization_commits()
    
    print("步骤13: 汇总知识库")
    filter_other.aggregate_final_commits()
    
    print("所有处理步骤完成！")

if __name__ == "__main__":
    main()