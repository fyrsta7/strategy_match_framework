import os
import shutil

def delete_target_subfolders(base_path):
    """
    遍历 base_path 下的每个 hash 目录，删除其中名称包含 'baseline' 或 'bm_25' 的子文件夹。

    :param base_path: 目标父目录路径，例如 "/raid/zyw/llm_on_code/llm_on_code_optimization/result/rocksdb/modified_file"
    """
    if not os.path.exists(base_path):
        print(f"错误：目录 {base_path} 不存在！")
        return

    # 遍历 base_path 下的所有子目录（假设目录名为 hash 数字）
    for hash_dir in os.listdir(base_path):
        hash_dir_path = os.path.join(base_path, hash_dir)
        if not os.path.isdir(hash_dir_path):
            continue

        # 遍历 hash 目录下的所有子文件夹
        for subfolder in os.listdir(hash_dir_path):
            subfolder_path = os.path.join(hash_dir_path, subfolder)
            if os.path.isdir(subfolder_path) and ('baseline' in subfolder or 'bm_25' in subfolder):
                try:
                    print(f"删除目录：{subfolder_path}")
                    shutil.rmtree(subfolder_path)
                except Exception as e:
                    print(f"删除 {subfolder_path} 时发生错误: {e}")

def main():
    # 指定 modified_file 目录路径，根据实际环境修改该路径
    base_modified_file_dir = "/raid/zyw/llm_on_code/llm_on_code_optimization/result/rocksdb/modified_file"
    
    print(f"开始处理目录：{base_modified_file_dir}")
    delete_target_subfolders(base_modified_file_dir)
    print("处理完成！")

if __name__ == "__main__":
    main()
