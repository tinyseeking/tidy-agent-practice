import json
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import os
import logging

from llm.call_llm_struct import get_schema_json

# 加载环境变量,读取.env文件配置信息
load_dotenv()

# 获取千问API密钥,请用百炼API Key将下行替换。
# 如何获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
qwen_api_key = os.getenv("QWEN_API_KEY")
if not qwen_api_key:
    raise ValueError("缺少QWEN_API_KEY环境变量")
# 获取千问请求端口URL
qwen_base_url = os.getenv("QWEN_BASE_URL")
if not qwen_base_url:
    qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


client = OpenAI(
    # 下面两个参数的默认值来自环境变量，可以不加
    api_key=qwen_api_key,
    base_url=qwen_base_url,
)


def call_llm_extract_api(messages: list) -> dict[Any, Any] | Any:
    """调用模型API获取回复"""
    logging.info("request messages: %s", messages)

    # 调用OpenAI API
    response = client.chat.completions.create(
        model="qwen-plus-latest",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        temperature=0,  # 降低随机性，使输出更确定性
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    logging.info("response messages: %s", content)
    try:
        # 将JSON字符串解析为Python字典
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始响应内容: {content}")
        return {}


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


def main():
    """提取结构化信息演示"""
    # 构建包含JSON Schema的系统提示
    system_prompt = (
        "你是一个信息提取专家。请根据用户描述提取信息，并严格按照以下JSON Schema返回JSON对象：\n"
        f"{get_schema_json(CalendarEvent)}\n\n"
        "注意：\n"
        "- 只返回符合Schema的JSON对象，不添加额外内容\n"
        "- 所有必需字段必须包含且类型正确\n"
        "- 日期应提取为字符串格式"
    )

    # 构建请求消息
    input_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        }
    ]

    # 调用千问模型
    response_data = call_llm_extract_api(input_messages)
    print("模型返回的JSON字符串:")
    print(response_data)

    # 验证并解析响应
    try:
        event = CalendarEvent(**response_data)
        print("\n解析后的对象:")
        print(f"名称: {event.name}")
        print(f"日期: {event.date}")
        print(f"参与者: {', '.join(event.participants)}")
    except Exception as e:
        print(f"\n解析失败: {e}")
        print(f"原始响应: {response_data}")


# 示例使用
if __name__ == "__main__":
    main()
