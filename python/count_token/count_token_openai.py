# https://github.com/openai/tiktoken?tab=readme-ov-file
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# https://platform.openai.com/docs/models

import tiktoken
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# 作为输入的的TXT文件路径
FILE_PATH = config.root_path + "test/input.txt"  
# 使用的OpenAI模型名称
MODEL_NAME = 'gpt-4o'       

def count_tokens(file_path, model_name):
    try:
        # 根据模型名称获取编码器
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"模型名称 '{model_name}' 未找到。请检查模型名称是否正确。")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"文件 '{file_path}' 未找到。请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 编码文本并计算Token数量
    tokens = encoding.encode(text)
    num_tokens = len(tokens)

    print(f"文件 '{file_path}' 中的Token数量（模型：{model_name}）：{num_tokens}")

if __name__ == "__main__":
    count_tokens(FILE_PATH, MODEL_NAME)