# Please install OpenAI SDK first: `pip3 install openai`
import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量,读取.env文件配置信息
load_dotenv()

# 设置DeepSeek API密钥
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise ValueError("缺少DEEPSEEK_API_KEY环境变量")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")
if not deepseek_base_url:
    raise ValueError("缺少DEEPSEEK_BASE_URL环境变量")

client = OpenAI(
    # 若没有配置环境变量，请用deepseek API Key将下行替换为：api_key="sk-xxx",
    # 获取deepseek key 地址： https://platform.deepseek.com/api_keys
    api_key=deepseek_api_key,
    base_url=deepseek_base_url
)

response = client.chat.completions.create(
    model="deepseek-chat",  # "deepseek-r1"
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "介绍下自己"},
    ],
    stream=False
)

print(response.choices[0].message.content)
