#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 全局配置
TARGET_FILE = "is_opt_keyword_arch_all.json"  # 要删除的目标文件名
DRY_RUN = False                         # 设置为True时只显示将要删除的文件
knowledge_base_dir = "knowledge_base"

def get_repo_list():
    """获取所有代码库列表"""
    try:
        # 直接从config导入知识库路径
        kb_path = Path(config.root_path) / knowledge_base_dir
        if not kb_path.exists():
            print(f"错误: 知识库路径不存在: {kb_path}")
            return None
        
        return kb_path, [d for d in os.listdir(kb_path) 
                        if (kb_path / d).is_dir()]
    except Exception as e:
        print(f"加载配置出错: {str(e)}")
        return None

def delete_target_files(kb_path, repos):
    """删除目标文件并返回统计结果"""
    stats = {'deleted': 0, 'errors': 0, 'skipped': 0}
    
    for repo in repos:
        target_path = kb_path / repo / TARGET_FILE
        
        if not target_path.exists():
            stats['skipped'] += 1
            continue
            
        try:
            if DRY_RUN:
                print(f"[模拟] 将删除: {target_path}")
            else:
                if target_path.is_file():
                    target_path.unlink()
                else:  # 如果是目录(虽然不太可能)
                    shutil.rmtree(target_path)
                print(f"✓ 已删除: {target_path}")
                stats['deleted'] += 1
        except Exception as e:
            print(f"✕ 删除失败 [{target_path}]: {str(e)}")
            stats['errors'] += 1
    
    return stats

def main():
    print(f"开始删除各代码库中的文件: {TARGET_FILE}")
    if DRY_RUN:
        print("⚠️ 当前为模拟模式，不会实际删除")
    
    # 获取知识库路径和代码库列表
    result = get_repo_list()
    if not result:
        sys.exit(1)
        
    kb_path, repos = result
    print(f"找到 {len(repos)} 个代码库")
    
    # 执行删除操作
    stats = delete_target_files(kb_path, repos)
    
    # 打印统计结果
    print("\n操作统计:")
    print(f"成功删除: {stats['deleted']}")
    print(f"删除失败: {stats['errors']}")
    print(f"跳过(文件不存在): {stats['skipped']}")
    if DRY_RUN:
        print("模拟模式已结束，实际未删除任何文件")

if __name__ == "__main__":
    # 添加项目根目录到Python路径
    sys.path.append(str(Path(__file__).parent.parent))
    import config  # 现在可以安全导入
    
    main()