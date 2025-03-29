import requests
import json
import os
import sys
import config

def get_models():
    url = config.xmcp_base_url + "/models"
    headers = {
        "Authorization": f"Bearer {config.xmcp_api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 如果响应状态码不是200，会抛出异常
    except requests.exceptions.RequestException as e:
        print("请求失败：", e)
        return None

    data = response.json()
    return data

def main():
    models_data = get_models()
    if models_data is None:
        print("获取模型列表失败。")
        return
    # 模型信息一般存储在 data 字段中
    models = models_data.get("data", [])
    print(f"共获取到 {len(models)} 个模型：")
    for model in models:
        # 输出模型名称以及其他你关心的信息
        print(f"- {model.get('id', 'Unknown')}")
    
    # 可选：将结果保存至文件
    output_file = "models_list.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(models_data, f, indent=4, ensure_ascii=False)
    print(f"\n模型列表已保存到 {output_file}")

if __name__ == "__main__":
    main()
