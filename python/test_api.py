from openai import OpenAI
import config

client = OpenAI(
    api_key = config.xmcp_api_key,
    base_url = config.xmcp_base_url
)

response = client.chat.completions.create(
    model=config.xmcp_claude_model,
    messages=[
        {
            'role': 'system',
            'content': "You are a helpful assistant."
        },
        {
            'role': 'user', 
            'content': "我应该如何在一个x86架构的ubuntu服务器上使用docker来用arm64编译redis？请你给我一个明确的方案"
        }
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='')
