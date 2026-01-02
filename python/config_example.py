import os

# ============================================================
# GitHub Token 配置
# ============================================================
# 【必须配置】GitHub Token 用于访问 GitHub API 获取代码库信息和 Commit 数据
# 获取方式：
#   1. 登录 GitHub，进入 Settings -> Developer Settings -> Personal access tokens -> Tokens (classic)
#   2. 点击 "Generate new token (classic)"
#   3. 设置 Token 名称（如 "SemOpt Access"）和过期时间
#   4. 勾选权限：至少需要 "repo" 权限
#   5. 生成后复制 Token（形如 ghp_xxxxxxxxxxxx），粘贴到下方 GITHUB_TOKEN 变量中
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = ""  # 请将你的 GitHub Token 粘贴到这里，替换空字符串
headers = {"Authorization": GITHUB_TOKEN}

# ============================================================
# 项目路径配置（自动获取，无需修改）
# ============================================================
# 该项目根目录，自动获取 huawei_stage2/ 的绝对路径
_config_file_path = os.path.abspath(__file__)
_python_dir = os.path.dirname(_config_file_path)
root_path = os.path.dirname(_python_dir) + "/"

# ============================================================
# LLM API 配置
# ============================================================
# 语言模型 API 用于执行代码优化、策略总结等任务
#
# 配置说明：
#   - xmcp_base_url: API 服务的基础 URL（通常以 /v1 结尾）
#   - xmcp_api_key: 你的 API Key（通常以 sk- 开头）
#   - xmcp_api_key_unlimit: 对使用的限制较少的 API Key（若无需区分，则设置为和 xmcp_api_key 相同）
#   - xmcp_deepseek_model: 使用的模型名称（如 "volc/deepseek-v3-250324"）

xmcp_base_url = ""
xmcp_api_key = ""
xmcp_api_key_unlimit = ""
xmcp_deepseek_model = ""
