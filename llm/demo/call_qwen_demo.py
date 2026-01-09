import os

from dotenv import load_dotenv
from openai import OpenAI


# 加载环境变量,读取.env文件配置信息
load_dotenv()

# 获取千问API密钥，请用百炼API Key将下行替换。
# 如何获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
qwen_api_key = os.getenv("QWEN_API_KEY")
if not qwen_api_key:
    raise ValueError("缺少QWEN_API_KEY环境变量")
# 获取千问请求端口URL
qwen_base_url = os.getenv("QWEN_BASE_URL")
if not qwen_base_url:
    qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


client = OpenAI(
    api_key=qwen_api_key,
    base_url=qwen_base_url,
)

completion = client.chat.completions.create(
    model="qwen-plus-latest",# 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '介绍下自己？'}
        ]
)
print(completion.choices[0].message.model_dump())
print(completion.choices[0].message.content)