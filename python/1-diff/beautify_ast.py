import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def beautify_ast(input_path, output_path):
    """
    读取 clang-diff 输出的 AST JSON 文件，对其进行格式化美化，
    并写入到输出文件中。
    """
    if not os.path.exists(input_path):
        print(f"输入文件不存在：{input_path}")
        return

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载 JSON 文件时出错：{e}")
        return

    # 利用 json.dumps 指定 indent 参数进行缩进格式化，
    # ensure_ascii=False 使得中文字符不会被转义
    pretty_json = json.dumps(data, indent=4, ensure_ascii=False)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_json)
        print(f"格式化完成，结果写入: {output_path}")
    except Exception as e:
        print(f"写入文件时出错：{e}")

file_name = "output_src"
input_path = os.path.join(config.root_path, "test", file_name + ".json")
output_path = os.path.join(config.root_path, "test", file_name + "_pretty.json")
beautify_ast(input_path, output_path)

file_name = "output_dst"
input_path = os.path.join(config.root_path, "test", file_name + ".json")
output_path = os.path.join(config.root_path, "test", file_name + "_pretty.json")
beautify_ast(input_path, output_path)