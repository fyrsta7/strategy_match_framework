import json
import shutil
import os
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def sync_back_repository_folders():
    """
    从repo_list_100.json中读取代码库信息，
    并将knowledge_base文件夹中对应的代码库文件夹同步回knowledge_base_all文件夹中
    """
    
    # 定义路径变量
    root_path = Path(config.root_path)
    repo_list_file = root_path / "repo_list_100.json"
    knowledge_base_dir = root_path / "knowledge_base"
    knowledge_base_all_dir = root_path / "knowledge_base_all"
    
    # 确保目标目录存在
    knowledge_base_all_dir.mkdir(exist_ok=True)
    
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
    not_found_count = 0
    
    # 使用tqdm显示进度条
    for repo in tqdm(repositories, desc="同步代码库文件夹回knowledge_base_all", unit="repo"):
        repo_name = repo.get("name")
        if not repo_name:
            print("警告: 发现没有名称的代码库，跳过")
            skipped_count += 1
            continue
        
        # 源文件夹（knowledge_base中的）和目标文件夹（knowledge_base_all中的）路径
        source_dir = knowledge_base_dir / repo_name
        target_dir = knowledge_base_all_dir / repo_name
        
        try:
            # 检查源文件夹是否存在
            if not source_dir.exists():
                print(f"警告: knowledge_base中不存在文件夹: {source_dir}")
                not_found_count += 1
                continue
            
            # 如果目标文件夹已存在，先删除
            if target_dir.exists():
                shutil.rmtree(target_dir)
                print(f"已删除knowledge_base_all中的现有文件夹: {target_dir}")
            
            # 复制文件夹
            shutil.copytree(source_dir, target_dir)
            success_count += 1
            
        except PermissionError:
            print(f"错误: 权限不足，无法同步 {repo_name}")
            failed_count += 1
        except shutil.Error as e:
            print(f"错误: 同步 {repo_name} 时发生错误: {e}")
            failed_count += 1
        except Exception as e:
            print(f"错误: 处理 {repo_name} 时发生未知错误: {e}")
            failed_count += 1
    
    # 输出统计结果
    print("\n" + "="*50)
    print("同步完成统计:")
    print(f"成功同步: {success_count} 个")
    print(f"同步失败: {failed_count} 个")
    print(f"未找到源文件夹: {not_found_count} 个")
    print(f"跳过处理: {skipped_count} 个")
    print(f"总计处理: {len(repositories)} 个")
    print("="*50)

def backup_existing_folders():
    """
    可选功能：在同步之前备份knowledge_base_all中已存在的文件夹
    """
    root_path = Path(config.root_path)
    repo_list_file = root_path / "repo_list_100.json"
    knowledge_base_all_dir = root_path / "knowledge_base_all"
    backup_dir = root_path / "knowledge_base_all_backup"
    
    # 读取JSON文件获取需要同步的代码库列表
    try:
        with open(repo_list_file, 'r', encoding='utf-8') as f:
            repositories = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取repo_list文件: {e}")
        return False
    
    # 创建备份目录
    backup_dir.mkdir(exist_ok=True)
    
    backup_count = 0
    for repo in tqdm(repositories, desc="备份现有文件夹", unit="repo"):
        repo_name = repo.get("name")
        if not repo_name:
            continue
            
        source_dir = knowledge_base_all_dir / repo_name
        backup_target = backup_dir / repo_name
        
        if source_dir.exists():
            try:
                if backup_target.exists():
                    shutil.rmtree(backup_target)
                shutil.copytree(source_dir, backup_target)
                backup_count += 1
            except Exception as e:
                print(f"警告: 备份 {repo_name} 失败: {e}")
    
    print(f"备份完成，共备份了 {backup_count} 个文件夹到 {backup_dir}")
    return True

if __name__ == "__main__":
    print("开始反向同步操作...")
    
    # 询问是否需要备份
    backup_choice = input("是否需要在同步前备份knowledge_base_all中的现有文件夹？(y/n): ").lower().strip()
    if backup_choice in ['y', 'yes']:
        print("正在备份现有文件夹...")
        if backup_existing_folders():
            print("备份完成，开始同步...")
        else:
            print("备份失败，是否继续同步？(y/n): ")
            if input().lower().strip() not in ['y', 'yes']:
                print("操作已取消")
                exit()
    
    # 执行同步操作
    sync_back_repository_folders()
    print("反向同步操作完成！")