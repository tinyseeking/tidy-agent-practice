import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量,读取.env文件配置信息
load_dotenv()

# 获取环境中智谱API密钥
api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("缺少ZHIPU_API_KEY环境变量")
base_url = os.getenv("ZHIPU_BASE_URL")
if not base_url:
    base_url = "https://open.bigmodel.cn/api/paas/v4/"

client = OpenAI(
    # 智谱AI大模型
    api_key=api_key,
    base_url=base_url
)

completion = client.chat.completions.create(
    model="glm-5",  # glm-5
    messages=[
        {"role": "system", "content": "你是一个聪明且富有创造力的小说作家"},
        {"role": "user",
         "content": "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"}
    ],
    top_p=0.7,
    temperature=0.9
)

print(completion.choices[0].message)
