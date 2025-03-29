#!/usr/bin/env python3
import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def remove_fields_from_json():
    """
    从 filtered_test_result.json 文件中删除每个 commit 的五个性能测试相关字段，
    然后将处理后的数据保存回原文件。
    """
    # 构建完整的文件路径
    json_file_path = os.path.join(
        config.root_path,
        'benchmark',
        'rocksdb',
        'filtered_test_result.json'
    )
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：文件 {json_file_path} 不存在!")
        return False
    
    try:
        # 读取原始 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"成功读取文件: {json_file_path}")
        print(f"开始处理 {len(data)} 条 commit 记录...")
        
        # 要删除的字段列表
        fields_to_remove = [
            'performance_test_child',
            'average_child',
            'performance_test_parent',
            'average_parent',
            'comparison_child_to_parent'
        ]
        
        # 计数
        processed_commits = 0
        removed_fields_count = 0
        
        # 处理每个 commit
        for commit in data:
            for field in fields_to_remove:
                if field in commit:
                    del commit[field]
                    removed_fields_count += 1
            processed_commits += 1
        
        # 将处理后的数据保存回原文件
        with open(json_file_path, 'w', encoding='utf-8') as file:
            # 使用 indent=4 保持良好的格式化，ensure_ascii=False 正确显示非ASCII字符
            json.dump(data, file, indent=4, ensure_ascii=False)
        
        print(f"处理完成!")
        print(f"- 处理的 commit 数: {processed_commits}")
        print(f"- 移除的字段数: {removed_fields_count}")
        print(f"- 已将结果保存到原文件: {json_file_path}")
        
        return True
        
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file_path} 不是有效的 JSON 格式!")
        return False
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return False

if __name__ == "__main__":
    remove_fields_from_json()