GITHUB_API_URL = "https://api.github.com"
# GitHub - Settings - Developer Settings - Personal access tokens - Tokens (classic) - 生成一个然后复制进来就行
GITHUB_TOKEN = ""
headers = {"Authorization": GITHUB_TOKEN}

# 该项目根目录，注意最后以 "/" 结尾
root_path = ""

# 获取方式：https://platform.deepseek.com/api_keys。没试过报销，应该可以吧？
deepseek_base_url = "https://api.deepseek.com"
deepseek_api_key = ""
deepseek_model = "deepseek-coder"

# 获取方式：https://www.closeai-asia.com/developer/api。其他 gpt 的 api 代理平台也可以，不过 CloseAI 组里可以报销。
chatgpt_base_url = "https://api.openai-proxy.org/v1"
chatgpt_api_key = ""
chatgpt_model = "gpt-4o"