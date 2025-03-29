import os
import json
import concurrent.futures
import tqdm
from pathlib import Path
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def process_json_file(file_path):
    """处理单个json文件，删除指定字段"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        fields_to_remove = ['modified_func_count', 'modified_other', 'modified_func']
        
        for commit in data:
            removed_fields = []
            for field in fields_to_remove:
                if field in commit:
                    del commit[field]
                    removed_fields.append(field)
                    modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            return file_path, True
        return file_path, False
    
    except Exception as e:
        return file_path, f"Error: {str(e)}"

def main():
    knowledge_base_dir = os.path.join(config.root_path, "knowledge_base")
    
    # 获取所有仓库目录
    repos = [d for d in os.listdir(knowledge_base_dir) 
             if os.path.isdir(os.path.join(knowledge_base_dir, d))]
    
    # 收集所有需要处理的文件
    files_to_process = []
    for repo in repos:
        has_file_json = os.path.join(knowledge_base_dir, repo, "has_file_with_func.json")
        if os.path.exists(has_file_json):
            files_to_process.append(has_file_json)
    
    # 使用进度条展示处理进度
    with tqdm.tqdm(total=len(files_to_process), desc="Processing files") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 提交任务
            future_to_file = {executor.submit(process_json_file, file_path): file_path 
                             for file_path in files_to_process}
            
            # 收集结果
            results = {"modified": 0, "not_modified": 0, "errors": 0}
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_path, result = future.result()
                    if result is True:
                        results["modified"] += 1
                    elif result is False:
                        results["not_modified"] += 1
                    else:
                        results["errors"] += 1
                        print(f"Error processing {file_path}: {result}")
                except Exception as e:
                    results["errors"] += 1
                    print(f"Error processing {file_path}: {str(e)}")
                pbar.update(1)
    
    # 打印统计结果
    print(f"Processing complete!")
    print(f"Files modified: {results['modified']}")
    print(f"Files not requiring modification: {results['not_modified']}")
    print(f"Files with errors: {results['errors']}")

if __name__ == "__main__":
    main()