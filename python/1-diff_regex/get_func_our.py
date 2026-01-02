import os
import json
import difflib
import re
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
import sys
import multiprocessing
import shutil
import glob
# 导入配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
# 全局配置
MAX_WORKERS = 128  # 并行处理的最大工作线程数
SKIP_PROCESSED = False  # 是否跳过已处理的commit
# 新增全局变量
KNOWLEDGE_BASE_ROOT = os.path.join(config.root_path, 'knowledge_base_all')
INPUT_JSON_FILENAME = 'diff.json'
OUTPUT_JSON_FILENAME = 'diff.json'
BEFORE_FUNC_FILENAME = 'before_func'
AFTER_FUNC_FILENAME = 'after_func'

def load_file(filepath):
    """加载文件内容并返回行列表"""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except:
        return []

def find_before_after_files(directory):
    """查找目录下的before.*和after.*文件"""
    try:
        before_files = glob.glob(os.path.join(directory, 'before.*'))
        after_files = glob.glob(os.path.join(directory, 'after.*'))
        if before_files and after_files:
            return before_files[0], after_files[0]
        else:
            return None, None
    except Exception:
        return None, None

def parse_functions(file_path):
    """
    解析 C/C++ 源文件，识别函数及其行范围。
    返回一个列表，包含字典 {'name': 函数名, 'start_line': 起始行, 'end_line': 结束行}。
    """
    functions = []
    # 更精确的函数签名正则模式 - 增强版，更好地处理跨行函数声明和复杂返回类型
    function_signature_pattern = re.compile(
        r'^[\s]*'                                     # 可能的前导空白
        r'(?:[\w:<>,~\*&]+\s+)*'                      # 返回类型（包括模板、指针、引用等）
        r'((?:[\w:]+::)?[\w~]+)'                      # 函数名（捕获组，包括类限定符）
        r'\s*\('                                      # 参数列表开始
    )
    # 构造函数模式 - 特别处理带有初始化列表的情况
    constructor_pattern = re.compile(
        r'^[\s]*'                                     # 可能的前导空白
        r'([\w~]+)'                                   # 构造函数名（捕获组）
        r'\s*\([^{;]*\)'                              # 参数列表（允许多行）
        r'(?:\s*:\s*[\w_]+\s*\([^)]*\).*)?'           # 可选的初始化列表
    )
    # 单行函数模式（包含开括号和闭括号）
    single_line_func_pattern = re.compile(
        r'^[\s]*'                                     # 可能的前导空白
        r'(?:[\w:<>,~\*&]+\s+)*'                      # 返回类型（包括模板、指针、引用等）
        r'((?:[\w:]+::)?[\w~]+)'                      # 函数名（捕获组，包括类限定符）
        r'\s*\([^)]*\)'                               # 参数列表
        r'(?:\s*const)?'                              # 可选的 'const'
        r'\s*'                                        # 可选的空白
        r'(?:throw\s*\([^)]*\))?'                     # 可选的 'throw' 说明符
        r'\s*'                                        # 可选的空白
        r'(?:override\s*)?'                           # 可选的 'override'
        r'\s*'                                        # 可选的空白
        r'{[^}]*}'                                    # 整个函数体在同一行
    )
    # 排除判断所需的关键词模式
    exclusion_keywords = re.compile(r'\b(if|else|for|while|switch|do|catch)\s*\(')
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return functions  # 返回空列表，避免中断程序
    except Exception:
        return functions
    i = 0
    in_function_declaration = False
    current_func_name = None
    declaration_start_line = 0
    paren_balance = 0
    while i < len(lines):
        line = lines[i].strip()
        # 跳过空行和预处理指令
        if not line or line.startswith('#'):
            i += 1
            continue
        # 如果正在处理函数声明
        if in_function_declaration:
            # 计算括号平衡
            paren_balance += line.count('(') - line.count(')')
            # 如果找到左大括号，说明函数声明结束
            if '{' in line and paren_balance <= 0:
                in_function_declaration = False
                # 寻找匹配的右花括号以找到函数体结束
                brace_count = line.count('{') - line.count('}')
                j = i
                if brace_count == 0:  # 函数体在同一行
                    functions.append({
                        'name': current_func_name,
                        'start_line': declaration_start_line,
                        'end_line': i + 1  # 从1开始计数行号
                    })
                    i += 1
                    continue
                j += 1
                while j < len(lines) and brace_count > 0:
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                if brace_count == 0:
                    functions.append({
                        'name': current_func_name,
                        'start_line': declaration_start_line,
                        'end_line': j
                    })
                i = j
                continue
        else:
            # 先检查是否是单行函数
            single_line_match = single_line_func_pattern.search(lines[i])
            if single_line_match and not exclusion_keywords.search(lines[i]):
                func_name = single_line_match.group(1)
                functions.append({
                    'name': func_name,
                    'start_line': i + 1,  # 从1开始计数行号
                    'end_line': i + 1     # 单行函数，起止行相同
                })
                i += 1
                continue
            # 检查是否是构造函数 - 专门处理构造函数
            constructor_match = constructor_pattern.search(lines[i])
            if constructor_match and not exclusion_keywords.search(lines[i]):
                # 向前看几行，确认这是构造函数定义而不是声明
                is_definition = False
                look_ahead_limit = 10
                for look_ahead in range(look_ahead_limit):
                    if i + look_ahead >= len(lines):
                        break
                    if '{' in lines[i + look_ahead]:
                        is_definition = True
                        break
                    if ';' in lines[i + look_ahead] and lines[i + look_ahead].strip().endswith(';'):
                        break
                if is_definition:
                    current_func_name = constructor_match.group(1)
                    declaration_start_line = i + 1
                    in_function_declaration = True
                    # 处理可能存在的初始化列表
                    j = i
                    while j < len(lines) and '{' not in lines[j]:
                        j += 1
                    if j < len(lines) and '{' in lines[j]:
                        # 找到函数体开始
                        brace_count = lines[j].count('{') - lines[j].count('}')
                        if brace_count == 0:  # 函数体在同一行内结束
                            functions.append({
                                'name': current_func_name,
                                'start_line': declaration_start_line,
                                'end_line': j + 1
                            })
                            in_function_declaration = False
                            i = j + 1
                            continue
                        # 否则继续寻找函数体结束位置
                        j += 1
                        while j < len(lines) and brace_count > 0:
                            brace_count += lines[j].count('{') - lines[j].count('}')
                            j += 1
                        if brace_count == 0:
                            functions.append({
                                'name': current_func_name,
                                'start_line': declaration_start_line,
                                'end_line': j
                            })
                            in_function_declaration = False
                            i = j
                            continue
            # 检查一般函数声明
            match = function_signature_pattern.search(lines[i])
            if match and not exclusion_keywords.search(lines[i]):
                # 检查是否真的是函数定义而不是函数声明或前向声明
                is_definition = False
                semicolon_found = False
                for look_ahead in range(10):  # 向前看10行
                    if i + look_ahead >= len(lines):
                        break
                    ahead_line = lines[i + look_ahead]
                    if '{' in ahead_line:
                        is_definition = True
                        break
                    if ';' in ahead_line and ahead_line.strip().endswith(';'):
                        semicolon_found = True
                        break
                # 只有确认是函数定义时才进行处理
                if is_definition and not semicolon_found:
                    current_func_name = match.group(1)
                    declaration_start_line = i + 1  # 从1开始计数行号
                    in_function_declaration = True
                    paren_balance = lines[i].count('(') - lines[i].count(')')
                    # 如果同一行已经包含函数体开始标记
                    if '{' in lines[i] and paren_balance <= 0:
                        in_function_declaration = False
                        # 寻找匹配的右花括号以找到函数体结束
                        brace_count = lines[i].count('{') - lines[i].count('}')
                        j = i
                        # 如果函数体在同一行内结束
                        if brace_count == 0:
                            functions.append({
                                'name': current_func_name,
                                'start_line': declaration_start_line,
                                'end_line': i + 1
                            })
                            i += 1
                            continue
                        j += 1
                        while j < len(lines) and brace_count > 0:
                            brace_count += lines[j].count('{') - lines[j].count('}')
                            j += 1
                        if brace_count == 0:
                            functions.append({
                                'name': current_func_name,
                                'start_line': declaration_start_line,
                                'end_line': j
                            })
                        i = j
                        continue
        i += 1
    return functions

def find_function_for_diff(functions, diff_line_index):
    """
    在函数列表中找到包含指定行号的函数
    返回函数名，如果没有找到则返回None
    """
    for func in functions:
        start = func['start_line']
        end = func['end_line']
        if start <= diff_line_index <= end:
            return func['name']
    return None

def analyze_diff(file1, file2):
    """分析两个文件的差异，并统计修改的函数情况"""
    # 加载文件内容
    old_lines = load_file(file1)
    new_lines = load_file(file2)
    # 解析两个文件中的所有函数
    functions_before = parse_functions(file1)
    functions_after = parse_functions(file2)
    # 使用 SequenceMatcher 获取两份文件的差异区块
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = sm.get_opcodes()
    # 统计数据
    modifications = []  # 记录每处修改
    modified_functions = defaultdict(lambda: {'insert': 0, 'delete': 0, 'replace': 0, 'modifications': []})
    mod_id = 0
    for tag, i1, i2, j1, j2 in opcodes:
        # 如果区块相同则跳过
        if tag == 'equal':
            continue
        mod_id += 1
        func_name = None
        # 根据不同修改类型选择参考旧版或新版来确定所属函数
        if tag in ('insert', 'replace'):
            # 使用新文件中的行号来确定函数
            func_name = find_function_for_diff(functions_after, j1 + 1)  # +1 因为parse_functions中是从1开始计数的
        elif tag == 'delete':
            # 使用旧文件中的行号来确定函数
            func_name = find_function_for_diff(functions_before, i1 + 1)  # +1 同上
        # 创建修改记录
        modification = {
            "id": mod_id,
            "type": tag,  # 'replace', 'insert', 或 'delete'
            "old_start": i1 + 1,  # 转换为1-based行号
            "old_end": i2 + 1,    # 转换为1-based行号并+1适应difflib的区间表示
            "new_start": j1 + 1,  # 转换为1-based行号
            "new_end": j2 + 1,    # 转换为1-based行号并+1适应difflib的区间表示
            "function": func_name,
            "old_content": "".join(old_lines[i1:i2]) if i1 < i2 else None,
            "new_content": "".join(new_lines[j1:j2]) if j1 < j2 else None
        }
        modifications.append(modification)
        # 更新函数修改统计
        if func_name:
            modified_functions[func_name][tag] += 1
            modified_functions[func_name]['modifications'].append(mod_id)
    # 计算不属于任何函数的修改
    non_function_mods = [mod["id"] for mod in modifications if mod["function"] is None]
    modified_funcs_list = list(modified_functions.keys())
    # 返回统计结果和函数信息
    return {
        "modified_func_count": len(modified_funcs_list),
        "modified_other": len(non_function_mods) > 0,
        "modified_func": modified_funcs_list,
        "functions_before": functions_before,
        "functions_after": functions_after,  # 添加修改后的函数列表
        "file_path_before": file1,
        "file_path_after": file2  # 添加修改后的文件路径
    }

def extract_function_content(file_path, func_name, functions):
    """提取函数的完整内容"""
    try:
        # 找到对应的函数
        target_func = None
        for func in functions:
            if func['name'] == func_name:
                target_func = func
                break
        if not target_func:
            return None
        # 获取函数的行范围
        start_line = target_func['start_line']
        end_line = target_func['end_line']
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        # 提取函数内容
        if start_line <= len(lines) and end_line <= len(lines):
            return ''.join(lines[start_line-1:end_line])  # 调整为0-based索引
        else:
            return None
    except Exception as e:
        print(f"提取函数内容时出错: {str(e)}")
        return None

def detect_file_extension(directory):
    """检测目录下before.*和after.*文件的后缀名"""
    try:
        # 查找before.*和after.*文件
        before_files = glob.glob(os.path.join(directory, 'before.*'))
        after_files = glob.glob(os.path.join(directory, 'after.*'))
        # 提取所有后缀名
        extensions = []
        for file_path in before_files + after_files:
            filename = os.path.basename(file_path)
            if '.' in filename:
                ext = filename.split('.', 1)[1]  # 获取第一个点后的所有内容作为后缀
                extensions.append(ext)
        if extensions:
            # 返回最常见的后缀名
            from collections import Counter
            most_common_ext = Counter(extensions).most_common(1)[0][0]
            return most_common_ext
        else:
            return 'txt'  # 默认后缀
    except Exception:
        return 'txt'  # 出错时返回默认后缀

def process_commit(repo_name, commit):
    """处理单个commit并分析函数修改，返回处理结果和状态信息"""
    commit_hash = commit["hash"]
    
    # 如果已经处理过且配置为跳过，则直接返回commit和跳过状态
    if SKIP_PROCESSED and "modified_func_count" in commit:
        return commit, {
            'skipped': True,
            'generated_files': False,
            'reason': '已处理，跳过'
        }
    
    # 对于每个修改的文件，分析前后差异
    modified_files = commit.get("modified_files", [])
    modified_funcs_all = set()
    has_non_func_changes = False
    modified_file_results = {}
    
    for file_name in modified_files:
        # 查找before.*和after.*文件
        commit_dir = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name, 'modified_file', commit_hash)
        before_path, after_path = find_before_after_files(commit_dir)
        
        # 确保文件存在
        if not before_path or not after_path:
            continue
        
        # 分析文件差异
        result = analyze_diff(before_path, after_path)
        
        # 收集结果
        modified_funcs_all.update(result["modified_func"])
        if result["modified_other"]:
            has_non_func_changes = True
        
        # 保存文件分析结果，用于后续提取函数内容
        modified_file_results[file_name] = result
    
    # 更新commit信息
    modified_commit = commit.copy()
    modified_commit["modified_func_count"] = len(modified_funcs_all)
    modified_commit["modified_other"] = has_non_func_changes
    modified_commit["modified_func"] = list(modified_funcs_all)
    
    # 初始化状态信息
    status_info = {
        'skipped': False,
        'generated_files': False,
        'reason': '',
        'func_name': None,
        'before_file_created': False,
        'after_file_created': False
    }
    
    # 如果只修改了一个函数，且没有非函数修改，则提取该函数修改前和修改后的完整内容
    if len(modified_funcs_all) == 1 and not has_non_func_changes:
        func_name = list(modified_funcs_all)[0]
        status_info['func_name'] = func_name
        
        # 创建输出目录
        output_dir = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name, 'modified_file', commit_hash)
        os.makedirs(output_dir, exist_ok=True)
        
        # 检测文件后缀名
        file_extension = detect_file_extension(output_dir)
        
        # 提取修改后的函数内容并保存
        for file_name, result in modified_file_results.items():
            if func_name in result["modified_func"]:
                # 从after.*中提取修改后的函数内容
                after_file_path = result["file_path_after"]
                functions_after = result["functions_after"]
                after_content = extract_function_content(after_file_path, func_name, functions_after)
                if after_content:
                    after_output_file = os.path.join(output_dir, f'{AFTER_FUNC_FILENAME}.{file_extension}')
                    with open(after_output_file, 'w', encoding='utf-8') as f:
                        f.write(after_content)
                    status_info['after_file_created'] = True
                
                # 从before.*中提取修改前的函数内容
                before_file_path = result["file_path_before"]
                functions_before = result["functions_before"]
                before_content = extract_function_content(before_file_path, func_name, functions_before)
                if before_content:
                    before_output_file = os.path.join(output_dir, f'{BEFORE_FUNC_FILENAME}.{file_extension}')
                    with open(before_output_file, 'w', encoding='utf-8') as f:
                        f.write(before_content)
                    status_info['before_file_created'] = True
                
                # 如果成功提取了内容，跳出循环
                if after_content or before_content:
                    break
        
        # 更新状态信息
        if status_info['before_file_created'] or status_info['after_file_created']:
            status_info['generated_files'] = True
            status_info['reason'] = f'单函数修改，已生成函数文件 ({func_name})'
        else:
            status_info['reason'] = f'单函数修改，但未能提取函数内容 ({func_name})'
    else:
        # 不满足生成文件的条件
        if len(modified_funcs_all) == 0:
            status_info['reason'] = '未检测到函数修改'
        elif len(modified_funcs_all) > 1:
            status_info['reason'] = f'修改了多个函数 ({len(modified_funcs_all)}个)'
        elif has_non_func_changes:
            status_info['reason'] = '除函数外还有其他修改'
    
    return modified_commit, status_info

def process_repository(repo_name):
    # 检查输入文件是否存在
    json_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name, INPUT_JSON_FILENAME)
    # 检查文件是否存在，不存在则跳过该代码库并给出提示
    if not os.path.exists(json_path):
        print(f"[{repo_name}] 输入文件不存在，跳过处理: {json_path}")
        return False
    
    try:
        # 加载commit数据
        with open(json_path, 'r', encoding='utf-8') as f:
            commits = json.load(f)
        
        print(f"[{repo_name}] 开始处理 {len(commits)} 个commits...")
        
        # 处理每个commit（顺序处理同一代码库中的commit）
        processed_commits = []
        for i, commit in enumerate(commits, 1):
            processed_commit, status_info = process_commit(repo_name, commit)
            processed_commits.append(processed_commit)
            
            # 输出处理状态
            commit_short = commit["hash"][:8]
            if status_info['skipped']:
                print(f"[{repo_name}] ({i}/{len(commits)}) {commit_short} - {status_info['reason']}")
            elif status_info['generated_files']:
                files_info = []
                if status_info['before_file_created']:
                    files_info.append(f"{BEFORE_FUNC_FILENAME}")
                if status_info['after_file_created']:
                    files_info.append(f"{AFTER_FUNC_FILENAME}")
                files_str = " + ".join(files_info) if files_info else "无"
                print(f"[{repo_name}] ({i}/{len(commits)}) {commit_short} - ✓ 已生成: {files_str} - {status_info['reason']}")
            else:
                print(f"[{repo_name}] ({i}/{len(commits)}) {commit_short} - × 未生成文件 - {status_info['reason']}")
        
        # 保存更新后的数据到输出文件
        output_path = os.path.join(KNOWLEDGE_BASE_ROOT, repo_name, OUTPUT_JSON_FILENAME)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_commits, f, indent=2, ensure_ascii=False)
        
        print(f"[{repo_name}] 处理完成，结果已保存到: {OUTPUT_JSON_FILENAME}")
        return True
    except Exception as e:
        print(f"[{repo_name}] 处理代码库时出错: {str(e)}")
        return False

def main():
    # 获取所有代码库 - 扫描knowledge_base目录下的所有子文件夹
    repositories = [dir_name for dir_name in os.listdir(KNOWLEDGE_BASE_ROOT)
                   if os.path.isdir(os.path.join(KNOWLEDGE_BASE_ROOT, dir_name))]
    print(f"找到 {len(repositories)} 个代码库，准备处理...")
    
    # 创建任务队列并显示进度条，不同代码库之间并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_repository, repo_name): repo_name for repo_name in repositories}
        
        # 创建代码库进度条
        repo_pbar = tqdm(total=len(repositories), desc="处理代码库")
        
        # 处理已完成的任务
        for future in concurrent.futures.as_completed(futures):
            repo_name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{repo_name}] 处理代码库时出错: {str(e)}")
            repo_pbar.update(1)
    
    repo_pbar.close()
    print("所有代码库处理完成。")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()