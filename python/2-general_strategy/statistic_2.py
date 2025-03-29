import json
import os
import csv
import sys

def compare_model_results():
    # Get file paths
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import config
    
    # Define file paths
    qwenmax_file_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_1000/partial_commit_1000_qwenmax.json")
    dsv3_file_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_1000/partial_commit_1000_dsv3.json")
    
    # Define output CSV file path
    output_file_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_1000/model_comparison_results.csv")
    
    # Load JSON files
    try:
        with open(qwenmax_file_path, 'r', encoding='utf-8') as f:
            qwenmax_commits = json.load(f)
        
        with open(dsv3_file_path, 'r', encoding='utf-8') as f:
            dsv3_commits = json.load(f)
            
        print(f"Files loaded: QwenMax file contains {len(qwenmax_commits)} commits, DSV3 file contains {len(dsv3_commits)} commits")
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    # Create hash-to-commit mappings for efficient lookup
    qwenmax_commits_dict = {commit['hash']: commit for commit in qwenmax_commits}
    dsv3_commits_dict = {commit['hash']: commit for commit in dsv3_commits}
    
    # Find commits common to both files
    common_hashes = set(qwenmax_commits_dict.keys()) & set(dsv3_commits_dict.keys())
    print(f"Found {len(common_hashes)} commits that exist in both files")
    
    # Write to CSV file
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write CSV header
            csv_writer.writerow([
                'hash', 
                'repository_name', 
                'qwenmax_optimization_summary_final',
                'qwenmax_is_generic_optimization_final', 
                'dsv3_optimization_summary_final',
                'dsv3_is_generic_optimization_final'
            ])
            
            # Write data for each common commit
            for hash_value in common_hashes:
                qwenmax_commit = qwenmax_commits_dict[hash_value]
                dsv3_commit = dsv3_commits_dict[hash_value]
                
                # Write a row
                csv_writer.writerow([
                    hash_value,
                    qwenmax_commit.get('repository_name', ''),
                    qwenmax_commit.get('optimization_summary_final', ''),
                    qwenmax_commit.get('is_generic_optimization_final', ''),
                    dsv3_commit.get('optimization_summary_final', ''),
                    dsv3_commit.get('is_generic_optimization_final', '')
                ])
                
        print(f"Comparison results successfully written to: {output_file_path}")
            
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    compare_model_results()