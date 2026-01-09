import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量,读取.env文件配置信息
load_dotenv()

# 国内访问openai需配置中转的 API地址或设置代理
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("缺少OPENAI_API_KEY环境变量")
base_url = os.getenv("OPENAI_BASE_URL")
if not base_url:
    raise ValueError("缺少OPENAI_BASE_URL环境变量")


def openai_chat():
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    # 创建一个聊天完成请求
    completion = client.chat.completions.create(
        # 指定模型为"gpt-5-turbo"
        model="gpt-5-turbo",
        # 定义对话消息列表
        messages=[
            # 系统角色的消息，用于设置对话的起始状态
            {"role": "system", "content": "You are a helpful assistant."},
            # 用户角色的消息，用于指示用户的输入
            {"role": "user", "content": "请写一首七言绝句, 描述夕阳"}
        ]
    )
    print(completion)  # 响应
    print(completion.choices[0].message)  # 回答


if __name__ == '__main__':
    openai_chat()