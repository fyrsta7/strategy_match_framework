import json
import os
import csv
import sys

def compare_commits():
    # Get file paths
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import config
    
    # Define all file paths
    file1_1_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_50_new/partial_commit_50_dsv3_1_1.json")
    file2_1_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_50_new/partial_commit_50_dsv3_2_1.json")
    file1_2_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_50_new/partial_commit_50_dsv3_1_2.json")
    file2_2_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_50_new/partial_commit_50_dsv3_2_2.json")
    
    # Define output CSV file path
    output_file_path = os.path.join(config.root_path, "python/2-general_strategy/result/partial_commit_50_new/comparison_results.csv")
    
    # Load all JSON files
    try:
        with open(file1_1_path, 'r', encoding='utf-8') as f:
            commits1_1 = json.load(f)
        
        with open(file2_1_path, 'r', encoding='utf-8') as f:
            commits2_1 = json.load(f)
            
        with open(file1_2_path, 'r', encoding='utf-8') as f:
            commits1_2 = json.load(f)
        
        with open(file2_2_path, 'r', encoding='utf-8') as f:
            commits2_2 = json.load(f)
            
        print(f"Files loaded: File 1_1 contains {len(commits1_1)} commits, File 2_1 contains {len(commits2_1)} commits")
        print(f"Files loaded: File 1_2 contains {len(commits1_2)} commits, File 2_2 contains {len(commits2_2)} commits")
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    # Create hash-to-commit mappings for efficient lookup
    commits1_1_dict = {commit['hash']: commit for commit in commits1_1}
    commits2_1_dict = {commit['hash']: commit for commit in commits2_1}
    commits1_2_dict = {commit['hash']: commit for commit in commits1_2}
    commits2_2_dict = {commit['hash']: commit for commit in commits2_2}
    
    # Find commits common to all four files
    common_hashes = set(commits1_1_dict.keys()) & set(commits2_1_dict.keys()) & set(commits1_2_dict.keys()) & set(commits2_2_dict.keys())
    print(f"Found {len(common_hashes)} commits that exist in all files")
    
    # Write to CSV file
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write CSV header
            csv_writer.writerow([
                'hash', 
                'repository_name', 
                'file1_1_optimization_summary_final',
                'file1_1_is_generic_optimization_final', 
                'file2_1_optimization_summary_final',
                'file2_1_is_generic_optimization_final',
                'file1_2_optimization_summary_final',
                'file1_2_is_generic_optimization_final', 
                'file1_2_is_strategy_equivalent_final', 
                'file2_2_optimization_summary_final',
                'file2_2_is_generic_optimization_final', 
                'file2_2_is_strategy_equivalent_final'
            ])
            
            # Write data for each common commit
            for hash_value in common_hashes:
                commit1_1 = commits1_1_dict[hash_value]
                commit2_1 = commits2_1_dict[hash_value]
                commit1_2 = commits1_2_dict[hash_value]
                commit2_2 = commits2_2_dict[hash_value]
                
                # Write a row
                csv_writer.writerow([
                    hash_value,
                    commit1_1.get('repository_name', ''),
                    commit1_1.get('optimization_summary_final', ''),
                    commit1_1.get('is_generic_optimization_final', ''),
                    commit2_1.get('optimization_summary_final', ''),
                    commit2_1.get('is_generic_optimization_final', ''),
                    commit1_2.get('optimization_summary_final', ''),
                    commit1_2.get('is_generic_optimization_final', ''),
                    commit1_2.get('is_strategy_equivalent_final', ''),
                    commit2_2.get('optimization_summary_final', ''),
                    commit2_2.get('is_generic_optimization_final', ''),
                    commit2_2.get('is_strategy_equivalent_final', '')
                ])
                
        print(f"Comparison results successfully written to: {output_file_path}")
            
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    compare_commits()