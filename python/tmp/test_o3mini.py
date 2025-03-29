import difflib
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 请在此处修改文件路径（旧版和新版）
FILE_NAME = "db/version_builder.cc"
OLD_FILE = "/home/zyw/llm_on_code/rocksdb/" + FILE_NAME
NEW_FILE = "/home/zyw/llm_on_code/rocksdb-2/" + FILE_NAME


def load_file(filepath):
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        return f.readlines()

def get_function_scope(lines, candidate_index):
    """
    从 candidate_index 所在行开始，寻找函数体的起始和结束行号（基于大括号匹配）。
    返回一个元组 (scope_start, scope_end)：
      - scope_start：函数体开始的行号（即第一个出现 '{' 的行）；
      - scope_end：大括号计数归零时所在的行号。
    若无法找到有效的大括号匹配，则返回 (None, None)。
    
    注意：此方法采用最简单的字符计数，未处理注释、字符串等特殊情况。
    """
    start = None
    line_index = candidate_index
    # 搜索函数体开始的 '{'
    while line_index < len(lines):
        line = lines[line_index]
        brace_pos = line.find('{')
        if brace_pos != -1:
            start = line_index
            break
        line_index += 1
    if start is None:
        return (None, None)
    
    count = 0
    scope_start = start
    scope_end = None
    # 从包含 '{' 的行开始计数
    for i in range(start, len(lines)):
        line = lines[i]
        count += line.count('{')
        count -= line.count('}')
        if count == 0:
            scope_end = i
            break
    return (scope_start, scope_end)

def find_function_for_diff(lines, diff_line_index):
    """
    从 diff_line_index 行向上扫描，寻找修改所在的函数定义。
    注意：如果修改发生在函数声明语句部分（即在函数体开始 '{' 之前），则不认为该修改归属于该函数，
    而继续向上寻找。只有当修改行位于某个函数体内时，才返回该函数名。
    
    匹配的正则适用于简单格式的C/C++函数定义，如：
      int foo(...){ 或
      void MyClass::bar(...) { 
    """
    # 正则模式较简单，可以根据需要增强（如处理模板、跨行声明等情况）
    pattern = re.compile(r"^\s*(?:[\w:<>\*\&\s]+)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{?")
    
    # 从 diff_line_index 往上扫描
    for i in range(diff_line_index, -1, -1):
        line = lines[i].strip()
        match = pattern.match(line)
        if match:
            func_name = match.group(1)
            scope_start, scope_end = get_function_scope(lines, i)
            if scope_start is None or scope_end is None:
                continue
            # 如果修改行位于函数体中，返回该函数名。
            # 注意：如果修改正好发生在函数声明的那一行（而函数体在后面），则不认为属于该函数，
            # 继续向上查找可能包含该修改的函数。
            if diff_line_index >= scope_start and diff_line_index <= scope_end:
                return func_name
    return None

def main():
    old_lines = load_file(OLD_FILE)
    new_lines = load_file(NEW_FILE)

    # 利用 SequenceMatcher 获取两份文件的差异区块
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = sm.get_opcodes()

    total_modifications = 0  # 按 diff 区块计算的修改总数
    modified_functions = {}  # 被修改函数字典，key 为函数名，value 为该函数对应的修改次数
    global_modifications = 0  # 不属于任何函数的修改计数
    details = []  # 保存每处修改的详细信息

    for tag, i1, i2, j1, j2 in opcodes:
        # 如果区块相同则跳过
        if tag == 'equal':
            continue

        total_modifications += 1
        func_name = None
        # 根据不同修改类型选择参考旧版或新版
        if tag in ('insert', 'replace'):
            if j1 < len(new_lines):
                func_name = find_function_for_diff(new_lines, j1)
        elif tag == 'delete':
            if i1 < len(old_lines):
                func_name = find_function_for_diff(old_lines, i1)

        if func_name:
            modified_functions[func_name] = modified_functions.get(func_name, 0) + 1
        else:
            global_modifications += 1

        details.append({
            "opcode": (tag, i1, i2, j1, j2),
            "function": func_name
        })

    # 输出整体对比结果
    print("文件修改情况比较：")
    print("旧文件：", OLD_FILE)
    print("新文件：", NEW_FILE)
    print("总共有 {} 处修改（按改动区块计算）。".format(total_modifications))
    print()

    print("每处修改归属的函数：")
    for idx, diff in enumerate(details, start=1):
        tag, i1, i2, j1, j2 = diff["opcode"]
        func_str = diff["function"] if diff["function"] else "【不属于任何函数】"
        print("  修改 {}: 区块类型：{}，归属函数：{}".format(idx, tag, func_str))
        
    print()
    print("总共有 {} 个函数内容被修改：".format(len(modified_functions)))
    if modified_functions:
        for fname, count in modified_functions.items():
            print("  函数 '{}' 修改了 {} 处".format(fname, count))
    else:
        print("  没有检测到任何修改归属于函数。")
    print("不属于任何函数的修改有 {} 处。".format(global_modifications))

if __name__ == "__main__":
    main()