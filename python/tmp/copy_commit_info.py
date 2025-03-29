import json
import os
import sys
import logging
from typing import List, Optional, Dict

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 定义JSON文件路径
ONE_FUNC_JSON_PATH = '/home/zyw/llm_on_code/llm_on_code_optimization/result/rocksdb/one_func.json'
TEST_RESULT_JSON_PATH = '/home/zyw/llm_on_code/llm_on_code_optimization/result/rocksdb/test_result.json'

# 要复制的字段
FIELDS_TO_COPY = ['repository_name', 'all_functions', 'modified_functions']

def load_json(file_path):
    """加载JSON文件"""
    if not os.path.exists(file_path):
        logging.error(f"文件不存在: {file_path}")
        sys.exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logging.info(f"成功加载文件: {file_path}")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误在文件 {file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"无法读取文件 {file_path}: {e}")
        sys.exit(1)

def save_json(data, file_path):
    """将数据保存到JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"成功保存文件: {file_path}")
    except Exception as e:
        logging.error(f"无法写入文件 {file_path}: {e}")
        sys.exit(1)

def index_commits_by_hash(commits: List[Dict], file_label: str) -> Dict[str, Dict]:
    """根据commit hash索引提交"""
    commit_dict = {}
    for commit in commits:
        commit_hash = commit.get('hash')
        if not commit_hash:
            logging.warning(f"在{file_label}中找到一个没有 'hash' 字段的提交，跳过。")
            continue
        if commit_hash in commit_dict:
            logging.warning(f"在{file_label}中找到重复的commit hash: {commit_hash}，仅保留第一个。")
            continue
        commit_dict[commit_hash] = commit
    return commit_dict

def merge_commits(one_func_commits: List[Dict], test_result_commits: List[Dict]) -> List[Dict]:
    """将one_func.json中的字段合并到test_result.json中对应的提交"""
    # 创建hash到commit的映射
    one_func_dict = index_commits_by_hash(one_func_commits, "one_func.json")
    logging.info(f"one_func.json中共有 {len(one_func_dict)} 个唯一的提交。")

    merged_count = 0
    skipped_count = 0
    not_found_count = 0

    for commit in test_result_commits:
        commit_hash = commit.get('hash')
        if not commit_hash:
            logging.warning("在test_result.json中找到一个没有 'hash' 字段的提交，跳过。")
            skipped_count += 1
            continue

        if commit_hash in one_func_dict:
            source_commit = one_func_dict[commit_hash]
            # 复制指定字段
            for field in FIELDS_TO_COPY:
                if field in source_commit:
                    commit[field] = source_commit[field]
                    logging.debug(f"复制字段 '{field}' 至commit {commit_hash}")
                else:
                    logging.warning(f"字段 '{field}' 在one_func.json的提交 {commit_hash} 中不存在。")
            merged_count += 1
        else:
            logging.warning(f"在one_func.json中未找到commit hash: {commit_hash}，跳过。")
            not_found_count += 1

    logging.info(f"合并完成：{merged_count} 个提交已更新，{skipped_count} 个提交被跳过，{not_found_count} 个提交在one_func.json中未找到。")
    return test_result_commits

def main():
    # 加载JSON数据
    one_func_data = load_json(ONE_FUNC_JSON_PATH)
    test_result_data = load_json(TEST_RESULT_JSON_PATH)

    if not isinstance(one_func_data, list):
        logging.error(f"one_func.json的内容不是列表格式，请检查文件结构。")
        sys.exit(1)

    if not isinstance(test_result_data, list):
        logging.error(f"test_result.json的内容不是列表格式，请检查文件结构。")
        sys.exit(1)

    # 合并提交数据
    updated_test_result = merge_commits(one_func_data, test_result_data)

    # 保存更新后的test_result.json
    save_json(updated_test_result, TEST_RESULT_JSON_PATH)

    logging.info("所有提交的合并操作已完成。")

if __name__ == "__main__":
    main()