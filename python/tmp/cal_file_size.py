import os
import json
import concurrent.futures
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Configuration variables
ROOT_PATH = config.root_path
MAX_WORKERS = 256  # 并行线程数，可以根据需要调整
JSON_FILE = "all_is_opt_final.json"  # JSON文件名称

def calculate_size_for_commit(commit):
    """Calculate the size of each file type for a single commit."""
    repo_name = commit["repository_name"]
    commit_hash = commit["hash"]
    
    # Define the path where the files are stored
    base_path = os.path.join(ROOT_PATH, "knowledge_base", repo_name, "modified_file", commit_hash)
    
    file_sizes = {
        "before.txt": 0,
        "before_func.txt": 0,
        "after.txt": 0,
        "after_func.txt": 0,
        "diff.txt": 0
    }
    
    # Check file existence and get sizes
    for file_name in file_sizes:
        file_path = os.path.join(base_path, file_name)
        if os.path.isfile(file_path):
            file_sizes[file_name] = os.path.getsize(file_path)
            
    return file_sizes

def main():
    # Path to the JSON file
    json_path = os.path.join(ROOT_PATH, JSON_FILE)
    
    # Load the JSON data
    print(f"Loading commits from {json_path}...")
    with open(json_path, 'r') as f:
        commits = json.load(f)
    
    print(f"Loaded {len(commits)} commits. Starting processing...")
    
    # Initialize the totals dictionary
    total_sizes = {
        "before.txt": 0,
        "before_func.txt": 0,
        "after.txt": 0,
        "after_func.txt": 0,
        "diff.txt": 0
    }
    
    # Process commits in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list to hold all futures
        futures = []
        
        # Submit all tasks
        for commit in commits:
            future = executor.submit(calculate_size_for_commit, commit)
            futures.append(future)
        
        # Process as they complete with a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Processing commits"):
            try:
                file_sizes = future.result()
                for file_name, size in file_sizes.items():
                    total_sizes[file_name] += size
            except Exception as e:
                print(f"An error occurred: {e}")
    
    # Convert sizes to megabytes for better readability
    total_sizes_mb = {file_name: size / (1024 * 1024) 
                     for file_name, size in total_sizes.items()}
    
    # Print the results
    print("\nTotal file sizes:")
    for file_name, size in total_sizes.items():
        print(f"{file_name}: {size} bytes ({total_sizes_mb[file_name]:.2f} MB)")
    
    # Calculate and print the grand total
    grand_total = sum(total_sizes.values())
    grand_total_mb = grand_total / (1024 * 1024)
    print(f"\nGrand total: {grand_total} bytes ({grand_total_mb:.2f} MB)")

if __name__ == "__main__":
    main()