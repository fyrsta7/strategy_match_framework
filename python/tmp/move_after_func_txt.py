import os
import shutil
import logging

# 配置日志记录
logging.basicConfig(
    filename='../../log/organize_optimization_results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置路径
ROOT_PATH = "/raid/zyw/llm_on_code/llm_on_code_optimization/result/rocksdb/modified_file"
MODEL_NAME = "deepseek-reasoner"  # 当前模型名，修改为需要的模型名

def organize_baseline_files(root_path: str, model_name: str):
    """
    遍历指定路径下的所有 commit 文件夹，检查 baseline 方案的优化结果文件，
    并将其移动到 baseline 子文件夹中。
    """
    if not os.path.exists(root_path):
        logging.error(f"路径不存在: {root_path}")
        print(f"路径不存在: {root_path}")
        return

    # 遍历所有 commit hash 文件夹
    for commit_hash in os.listdir(root_path):
        commit_path = os.path.join(root_path, commit_hash)

        # 检查是否是文件夹
        if not os.path.isdir(commit_path):
            logging.warning(f"跳过非文件夹路径: {commit_path}")
            continue

        # 构造 baseline 文件路径
        baseline_file = os.path.join(commit_path, f"after_func_{model_name}.txt")
        if os.path.exists(baseline_file):
            # 构造 baseline 子文件夹路径
            baseline_folder = os.path.join(commit_path, "baseline")

            # 创建 baseline 子文件夹（如果不存在）
            os.makedirs(baseline_folder, exist_ok=True)

            # 移动文件到 baseline 子文件夹
            target_file = os.path.join(baseline_folder, f"after_func_{model_name}.txt")
            try:
                shutil.move(baseline_file, target_file)
                logging.info(f"成功移动文件: {baseline_file} -> {target_file}")
                print(f"成功移动文件: {baseline_file} -> {target_file}")
            except Exception as e:
                logging.error(f"移动文件失败: {baseline_file} -> {target_file}, 错误: {e}")
                print(f"移动文件失败: {baseline_file} -> {target_file}, 错误: {e}")
        else:
            logging.info(f"未找到 baseline 方案的文件: {baseline_file}")
            print(f"未找到 baseline 方案的文件: {baseline_file}")

if __name__ == "__main__":
    # 调用函数
    organize_baseline_files(ROOT_PATH, MODEL_NAME)