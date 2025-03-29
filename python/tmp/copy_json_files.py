import os
import sys
import json
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 导入配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import config
except ImportError:
    print("无法导入config模块，请确保config.py文件存在")
    sys.exit(1)

# 需要处理的代码库列表
repositories = [
    "CMSIS_6",
    "Ditto",
    "oot",
    "kona",
    "bruteforce-wallet",
    "openMHA",
    "control_toolbox",
    "OpenCat",
    "RiderSourceCodeAccess",
    "interim",
    "TerminalMediaViewer",
    "manif",
    "pdfio",
    "Hello-World-in-Different-Languages",
    "uevloop",
    "apfsprogs",
    "json65",
    "tsl-sdr",
    "Apache-HTTP-Server-Module-Backdoor",
    "hacktoberfest2022",
    "msi-ec",
    "CppCmb",
    "ureq",
    "libPusher",
    "writexl",
    "trre"
]

def copy_json_file(repo_name):
    """
    将指定代码库的has_file.json文件复制到has_file_with_func.json
    """
    knowledge_base_dir = os.path.join(config.root_path, 'knowledge_base', repo_name)
    
    # 检查knowledge_base目录是否存在
    if not os.path.exists(knowledge_base_dir):
        print(f"[{repo_name}] 知识库目录不存在")
        return False
    
    source_file = os.path.join(knowledge_base_dir, 'has_file.json')
    target_file = os.path.join(knowledge_base_dir, 'has_file_with_func.json')
    
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        print(f"[{repo_name}] has_file.json 不存在")
        return False
    
    try:
        # 验证JSON文件格式
        with open(source_file, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                # 确保JSON内容是列表
                if not isinstance(json_data, list):
                    print(f"[{repo_name}] has_file.json 不是有效的JSON数组")
                    return False
            except json.JSONDecodeError:
                print(f"[{repo_name}] has_file.json 不是有效的JSON文件")
                return False
        
        # 复制文件
        shutil.copy2(source_file, target_file)
        print(f"[{repo_name}] 成功复制 has_file.json 到 has_file_with_func.json")
        return True
    except Exception as e:
        print(f"[{repo_name}] 复制文件时出错: {str(e)}")
        return False

def main():
    """主函数"""
    success_count = 0
    failure_count = 0
    
    print(f"开始处理 {len(repositories)} 个代码库...")
    
    for repo_name in repositories:
        if copy_json_file(repo_name):
            success_count += 1
        else:
            failure_count += 1
    
    print("\n处理完成!")
    print(f"成功: {success_count}")
    print(f"失败: {failure_count}")

if __name__ == "__main__":
    main()