import os
from typing import Any

import json

from dotenv import load_dotenv
import logging

from openai import OpenAI

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量,读取.env文件配置信息
load_dotenv()

"""
模型调用配置说明：
1. 默认接入阿里云百炼「通义千问」系列；可无缝切换到任何 OpenAI SDK格式的模型服务。（如GPT系列、智谱清言、deepseek等）
2. 使用千问模型，需配置的环境变量：
   - QWEN_API_KEY: 模型API请求密钥，这里使用阿里云百炼API密钥（获取地址：https://help.aliyun.com/zh/model-studio/get-api-key）
   - QWEN_BASE_URL: 模型API服务地址（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
3. 若用其它模型服务或自行部署的模型，按需替换api_key、base_url值即可。
"""
api_key = os.getenv("QWEN_API_KEY")
if not api_key:
    raise ValueError("缺少 QWEN_API_KEY 环境变量（从阿里云控制台获取）")
base_url = os.getenv("QWEN_BASE_URL")
if not base_url:
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 构建模型调用客户端，支持所有兼容openai SDK的模型服务,按需替换api_key、base_url值即可.
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# 使用和风天气API获取天气信息，文档地址：https://dev.qweather.com/docs/start/
# 需要配置环境变量 HEFENG_API_KEY(和风API密钥)、HEFENG_BASE_URL(开发者独立的API调用地址)
from tools.weather.weather_hefeng import search_city_info, get_current_weather


# 定义工具函数
def fetch_current_weather(location: str) -> dict:
    """获取指定城市当日的天气信息"""
    # 1. 搜索查询城市，获取城市ID
    city_search = search_city_info(query=location, number=2)
    logging.debug(f"city_search result :{city_search}")
    if not city_search.get("error") and city_search["results"]:
        location_id = city_search["results"][0]["location_id"]
        logging.debug(f"找到 ID: {location_id}")

        # 2. 根据城市ID获取当前天气
        curr_weather = get_current_weather(location_id)
        logging.debug(f"获取当前天气: {curr_weather}")
        return curr_weather
    return {"error": f"为查询到相关城市数据： location:{location}"}


def calculate(expression: str) -> float:
    """执行数学计算"""
    allowed_ops = ['+', '-', '*', '/', '**', '(', ')', '.']
    if any(char not in allowed_ops + [str(i) for i in range(10)] for char in expression):
        raise ValueError("表达式包含不安全字符")

    return eval(expression)


# 工具列表和对应的实现函数,方便根据函数名称执行调用函数
TOOLS = {
    "fetch_current_weather": fetch_current_weather,
    "calculate": calculate
}

# 工具定义（符合Function Calling规范,指定函数的名称、功能描述、参数及其类型说明）
FUNCTION_DEFINITIONS = [
    {
        "name": "fetch_current_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名称"},
            },
            "required": ["location"]
        },
        "description": "获取指定城市当日的天气信息"
    },
    {
        "name": "calculate",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expression"]
        },
        "description": "执行数学计算"
    }
]


def call_llm(messages: list, functions: list = None) -> dict[str, Any]:
    """调用千问API，支持Function Calling"""
    try:
        logging.info("request messages: %s", messages)

        # 调用参数定义
        # kwargs = {
        #     "model": "qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        #     "messages": messages,
        #     "temperature": 0,  # 降低随机性，使输出更确定性
        #     # "max_tokens": 2048
        # }
        #
        # # 如果提供了函数定义，则启用Function Calling
        # if functions:
        #     kwargs["functions"] = functions
        #
        # # 调用模型API
        # response = client.chat.completions.create(
        #     **kwargs
        # )
        # 向LLM发起请求
        response = client.chat.completions.create(
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            temperature=0,  # 降低随机性，使输出更确定性
            functions=functions,  # 启用Function Calling
            # max_tokens= 2048
        )

        logging.info("response messages: %s", response.choices[0].message)
        return response.model_dump()
    except Exception as e:
        raise Exception(f"API调用失败: {str(e)}")


def run_conversation(user_message: str) -> str:
    """运行完整的对话流程，使用Function Calling机制"""
    system_prompt = "你是一个智能助手，可以使用工具解决用户的问题。"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # 控制最大的执行工具调用次数
    max_function_calls = 5
    function_call_count = 0

    while function_call_count < max_function_calls:
        # 调用API，传入函数定义
        response = call_llm(messages, FUNCTION_DEFINITIONS)
        message = response["choices"][0]["message"]

        # 检查是否包含函数调用
        if "function_call" in message and message["function_call"]:
            function_call = message["function_call"]
            function_name = function_call["name"]
            parameters = json.loads(function_call["arguments"])

            # 记录函数调用
            messages.append({
                "role": "assistant",
                "function_call": function_call
            })

            # 执行函数调用
            if function_name in TOOLS:
                try:
                    function_to_call = TOOLS[function_name]
                    function_response = function_to_call(**parameters)
                    logging.info(f"调用工具：{function_name} 参数:{parameters}，执行结果: {function_response}")
                    # 添加函数执行结果到对话
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(function_response)
                    })

                    function_call_count += 1
                except Exception as e:
                    return f"执行工具时出错: {e}"
            else:
                return f"未知工具: {function_name}"
        else:
            # 没有函数调用，返回最终回答
            messages.append(message)
            return message["content"]

    return "已达到最大工具调用次数，无法继续处理。"


# 示例使用
if __name__ == "__main__":
    # 示例1：需要多轮工具调用的天气查询
    user_question1 = "上海天气如何？"
    print(f"用户: {user_question1}")
    answer1 = run_conversation(user_question1)
    print(f"助手: {answer1}")

    # 示例2：数学计算
    user_question2 = "5*5*1000*333是多少？"
    print(f"\n用户: {user_question2}")
    answer2 = run_conversation(user_question2)
    print(f"助手: {answer2}")
