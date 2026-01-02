import os

GITHUB_API_URL = "https://api.github.com"
# GitHub - Settings - Developer Settings - Personal access tokens - Tokens (classic) - 生成一个然后复制进来就行
GITHUB_TOKEN = "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"
headers = {"Authorization": GITHUB_TOKEN}

# 该项目根目录，自动获取 semopt_arch/ 的绝对路径
# config.py 位于 semopt_arch/python/config.py
# 向上两级得到 semopt_arch/ 的路径
_config_file_path = os.path.abspath(__file__)  # .../semopt_arch/python/config.py
_python_dir = os.path.dirname(_config_file_path)  # .../semopt_arch/python/
root_path = os.path.dirname(_python_dir) + "/"  # .../semopt_arch/

# semopt_c_paper_backup 和 semopt_arch 在同一个父目录下
_parent_dir = os.path.dirname(root_path.rstrip("/"))  # .../
semopt_c_paper_backup_path = os.path.join(_parent_dir, "semopt_c_paper_backup") + "/"

eval1_path = root_path + "eval1/"

# https://llm.xmcp.ltd/service_portal/
xmcp_base_url = "https://llm.xmcp.ltd/v1"
xmcp_api_key = "sk-4R8773oVtnZKF4S1r-HmKA"
xmcp_api_key_unlimit = "sk-29K4dbXPmvoN9T5Bb1nAWQ"
xmcp_gpt_model = "closeai/gpt-4o"
xmcp_o3_mini_model = "closeai/o3-mini"
xmcp_gpt_41_model = "yunwu/gpt-4.1-2025-04-14"
xmcp_deepseek_model = "volc/deepseek-v3-250324"
xmcp_qwen_model = "ali/qwen-max-latest"
xmcp_claude_model = "aws/claude-3-7-sonnet-20250219"

# 火山引擎，https://console.volcengine.com/
volc_base_url = "https://ark.cn-beijing.volces.com/api/v3"
volc_api_key = "bf695cfd-8c3b-4808-bcfb-fb48f7fc5bd5"
volc_deepseek_model = "deepseek-v3-250324"

# yunwu, https://yunwu.ai/console/token
yunwu_base_url = "https://yunwu.ai/v1"
yunwu_api_key = "sk-JZdaEtqm6gWMbJG7hQK1NYKdroRctXbqFTahQ56HsZvPMmmh"
yunwu_gemini_model = "gemini-2.5-pro-exp-03-25"

# 获取方式：https://platform.deepseek.com/api_keys。组内可以报销。
deepseek_base_url = "https://api.deepseek.com/v1"
deepseek_api_key = "sk-bb0b71b5541c498498e5a6fd992270da"
deepseek_model = "deepseek-chat"
deepseek_model = "deepseek-reasoner"

# SiliconFlow, https://cloud.siliconflow.cn/account/ak
siliconflow_base_url = "https://api.siliconflow.cn/v1"
siliconflow_api_key = "sk-oiqixsgwchfllkvzsafwzdoqtewuvwoddqakdgoboelanius"
siliconflow_deepseek_model = "deepseek-ai/DeepSeek-V3"
siliconflow_qwen_model = "Qwen/Qwen2.5-Coder-32B-Instruct"

# https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api
# https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.3b5d79807WSeUH#/home
# https://bailian.console.aliyun.com/?apiKey=1#/api-key
bailian_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
bailian_api_key = "sk-dd69975413e84b1a982bcce59f50e32c"
bailian_deepseek_model = "deepseek-v3"
bailian_qwen_model = "qwen-plus-2025-01-25"
bailian_llama_model = "llama3.3-70b-instruct"

# syc 学长给的api key，他那边可以报销
bailian_syc_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
bailian_syc_api_key = "sk-18fbfb3f138b4abba8c9b577d41db725"
bailian_syc_qwen_model = "qwen-turbo"

# 获取方式：https://www.closeai-asia.com/developer/api。其他 OpenAI 的 api 代理平台也可以，不过 CloseAI 组里可以报销。
closeai_base_url = "https://api.openai-proxy.org/v1"
closeai_api_key = "sk-dtxsT7a1n8U6n3YUwzq8jxAbWYD6N1ZXJHtNAWtyXL3nodp0"
closeai_chatgpt_model = "gpt-4o"
