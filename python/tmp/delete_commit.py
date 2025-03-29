import json
import sys

def load_json(file_path):
    """
    加载JSON文件并返回其内容。
    
    :param file_path: JSON文件的路径
    :return: JSON内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析JSON文件 - {file_path}\n详情: {e}")
        sys.exit(1)

def save_json(data, file_path):
    """
    将数据保存到JSON文件。
    
    :param data: 要保存的数据
    :param file_path: 目标文件路径
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"成功保存过滤后的数据到: {file_path}")
    except Exception as e:
        print(f"错误: 无法保存JSON文件 - {file_path}\n详情: {e}")
        sys.exit(1)

def extract_hashes(json_data, hash_key='hash'):
    """
    从JSON数据中提取所有的hash值。
    
    :param json_data: JSON数据（列表形式）
    :param hash_key: 用于标识hash值的键名
    :return: 包含所有hash值的集合
    """
    return {item[hash_key] for item in json_data if hash_key in item}

def filter_test_result(test_result, valid_hashes, hash_key='hash'):
    """
    过滤test_result中的提交，只保留那些存在于valid_hashes中的提交。
    
    :param test_result: 原始test_result数据（列表形式）
    :param valid_hashes: 有效的hash集合
    :param hash_key: 用于标识hash值的键名
    :return: 过滤后的test_result数据
    """
    filtered = [item for item in test_result if item.get(hash_key) in valid_hashes]
    removed = len(test_result) - len(filtered)
    print(f"已删除 {removed} 个不在one_func.json中的提交。")
    return filtered

def main():
    # 定义文件路径
    test_result_path = '/home/zyw/llm_on_code/llm_on_code_optimization/result/rocksdb/test_result.json'
    one_func_path = '/home/zyw/llm_on_code/llm_on_code_optimization/result/rocksdb/one_func.json'
    
    # 加载JSON文件
    print("加载test_result.json...")
    test_result = load_json(test_result_path)
    
    print("加载one_func.json...")
    one_func = load_json(one_func_path)
    
    # 提取one_func中的hash
    print("提取one_func.json中的hash...")
    valid_hashes = extract_hashes(one_func)
    print(f"找到 {len(valid_hashes)} 个有效的hash。")
    
    # 过滤test_result
    print("过滤test_result.json中的提交...")
    filtered_test_result = filter_test_result(test_result, valid_hashes)
    
    # 保存过滤后的数据回test_result.json
    save_json(filtered_test_result, test_result_path)
    
    print("过滤完成！")

if __name__ == "__main__":
    main()