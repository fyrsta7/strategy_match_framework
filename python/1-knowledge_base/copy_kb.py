import json
import shutil
import os
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def copy_repository_folders():
    """
    从repo_list_100.json中读取代码库信息，
    并将每个代码库的knowledge_base_all文件夹复制到knowledge_base文件夹中
    """
    
    # 定义路径变量
    root_path = Path(config.root_path)
    repo_list_file = root_path / "repo_list_100.json"
    knowledge_base_all_dir = root_path / "knowledge_base_all"
    knowledge_base_dir = root_path / "knowledge_base"
    
    # 确保目标目录存在
    knowledge_base_dir.mkdir(exist_ok=True)
    
    # 读取JSON文件
    try:
        with open(repo_list_file, 'r', encoding='utf-8') as f:
            repositories = json.load(f)
        print(f"成功读取到 {len(repositories)} 个代码库信息")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {repo_list_file}")
        return
    except json.JSONDecodeError:
        print(f"错误: JSON文件格式不正确 {repo_list_file}")
        return
    
    # 统计信息
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 使用tqdm显示进度条
    for repo in tqdm(repositories, desc="复制代码库文件夹", unit="repo"):
        repo_name = repo.get("name")
        if not repo_name:
            print("警告: 发现没有名称的代码库，跳过")
            skipped_count += 1
            continue
        
        # 源文件夹和目标文件夹路径
        source_dir = knowledge_base_all_dir / repo_name
        target_dir = knowledge_base_dir / repo_name
        
        try:
            # 检查源文件夹是否存在
            if not source_dir.exists():
                print(f"警告: 源文件夹不存在: {source_dir}")
                failed_count += 1
                continue
            
            # 如果目标文件夹已存在，先删除
            if target_dir.exists():
                shutil.rmtree(target_dir)
                print(f"已删除现有目标文件夹: {target_dir}")
            
            # 复制文件夹
            shutil.copytree(source_dir, target_dir)
            success_count += 1
            
        except PermissionError:
            print(f"错误: 权限不足，无法复制 {repo_name}")
            failed_count += 1
        except shutil.Error as e:
            print(f"错误: 复制 {repo_name} 时发生错误: {e}")
            failed_count += 1
        except Exception as e:
            print(f"错误: 处理 {repo_name} 时发生未知错误: {e}")
            failed_count += 1
    
    # 输出统计结果
    print("\n" + "="*50)
    print("复制完成统计:")
    print(f"成功复制: {success_count} 个")
    print(f"复制失败: {failed_count} 个")
    print(f"跳过处理: {skipped_count} 个")
    print(f"总计处理: {len(repositories)} 个")
    print("="*50)

if __name__ == "__main__":
    copy_repository_folders()