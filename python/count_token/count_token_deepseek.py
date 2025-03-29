# https://api-docs.deepseek.com/zh-cn/quick_start/token_usage

# pip3 install transformers
# python3 count_token_deepseek.py
import os
from transformers import AutoTokenizer

# 设置文件路径
input_file = os.path.join("./input.txt")

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "./",  # 确保分词器模型文件在此目录
    trust_remote_code=True
)

def count_tokens_from_file(file_path):
    try:
        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 计算token
        tokens = tokenizer.encode(content)
        token_strings = tokenizer.convert_ids_to_tokens(tokens)
        
        # 输出结果
        print(f"\n=== 文件内容（前100字符）===\n{content[:100]}...\n")
        print("Tokens列表:", tokens)
        print("\nTokens字符串表示:", token_strings)
        print(f"\n总Token数量: {len(tokens)}")
        
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    print(f"正在分析文件: {input_file}")
    count_tokens_from_file(input_file)