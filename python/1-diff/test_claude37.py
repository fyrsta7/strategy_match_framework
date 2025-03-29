import difflib
import re
import os
import json
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 文件路径设置
FILE_NAME = "db/c_test.c"

original_file = "/home/zyw/llm_on_code/rocksdb/" + FILE_NAME
modified_file = "/home/zyw/llm_on_code/rocksdb-2/" + FILE_NAME
output_json = "/raid/zyw/llm_on_code/llm_on_code_optimization/python/1-diff/output_claude37.json"
def load_file(filepath):
    """加载文件内容并返回行列表"""
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        return f.readlines()

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
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: 文件 {file_path} 不存在。")
        return functions  # 返回空列表，避免中断程序
    
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
            
            # 检查是否是构造函数 - 这是新增的代码，专门处理构造函数
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
    
    # 将函数信息转换为更标准的格式
    functions_before_dict = [
        {"name": f["name"], "start_line": f["start_line"], "end_line": f["end_line"]} 
        for f in functions_before
    ]
    
    functions_after_dict = [
        {"name": f["name"], "start_line": f["start_line"], "end_line": f["end_line"]} 
        for f in functions_after
    ]
    
    return modifications, modified_functions, functions_before_dict, functions_after_dict
def format_line_numbers(start, end):
    """格式化行号范围，用于显示"""
    if start == end - 1:
        return f"line {start}"
    else:
        return f"lines {start}-{end-1}"
def main():
    if not os.path.exists(original_file) or not os.path.exists(modified_file):
        print(f"错误：找不到文件 {original_file} 或 {modified_file}")
        return
    
    modifications, modified_functions, functions_before, functions_after = analyze_diff(original_file, modified_file)
    
    # 计算不属于任何函数的修改
    non_function_mods = [mod["id"] for mod in modifications if mod["function"] is None]
    
    # 创建要输出的JSON数据结构
    output_data = {
        "summary": {
            "original_file": original_file,
            "modified_file": modified_file,
            "total_modifications": len(modifications),
            "non_function_modifications": len(non_function_mods),
            "modified_functions_count": len(modified_functions)
        },
        "modifications": modifications,
        "modified_functions": {},
        "functions": {
            "original_file": functions_before,
            "modified_file": functions_after
        }
    }
    
    # 将defaultdict转换为标准字典格式
    for func_name, info in modified_functions.items():
        output_data["modified_functions"][func_name] = {
            "total_modifications": sum(v for k, v in info.items() if k != 'modifications'),
            "insert_count": info["insert"],
            "delete_count": info["delete"],
            "replace_count": info["replace"],
            "modification_ids": sorted(info["modifications"])
        }
    
    # 输出到JSON文件
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"分析结果已保存到 {output_json}")
    
    # 打印基本统计信息
    print(f"\n基本统计信息:")
    print(f"  总修改数: {len(modifications)}")
    print(f"  不属于任何函数的修改数: {len(non_function_mods)}")
    print(f"  被修改的函数数量: {len(modified_functions)}")
    
    # 打印函数列表和行号
    print("\n原始文件中的函数:")
    for func in sorted(functions_before, key=lambda x: x["start_line"]):
        print(f"  {func['name']}: 行 {func['start_line']}-{func['end_line']}")
    
    print("\n修改后文件中的函数:")
    for func in sorted(functions_after, key=lambda x: x["start_line"]):
        print(f"  {func['name']}: 行 {func['start_line']}-{func['end_line']}")
    
    # 输出每处修改详情
    print("\n每处修改详情：")
    for mod in modifications:
        func_info = f"在函数 '{mod['function']}' 中" if mod['function'] else "不在任何函数内"
        old_range = format_line_numbers(mod['old_start'], mod['old_end']) if mod['old_content'] else "N/A"
        new_range = format_line_numbers(mod['new_start'], mod['new_end']) if mod['new_content'] else "N/A"
        
        print(f"修改 #{mod['id']}:")
        print(f"  类型: {mod['type']}")
        print(f"  旧文件: {old_range}")
        print(f"  新文件: {new_range}")
        print(f"  位置: {func_info}")
        print()
    
    # 打印被修改的函数名称
    print("\n被修改的函数:")
    for i, func_name in enumerate(sorted(modified_functions.keys())):
        info = modified_functions[func_name]
        total_mods = sum(v for k, v in info.items() if k != 'modifications')
        print(f"  {i+1}. {func_name} (修改次数: {total_mods})")
    
    # 打印不属于任何函数的修改数量
    print(f"\n不属于任何函数的修改: {len(non_function_mods)} 处")
if __name__ == "__main__":
    main()