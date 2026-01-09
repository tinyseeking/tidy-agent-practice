import json
import os
import re
from typing import Dict, List

import requests
import urllib.parse

# 加载环境变量
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def initialize_model_client() -> OpenAI:
    """
     获取模型调用客户端,使用openai SDK，支持所有兼容openai接口的模型服务，默认使用千问模型

     Returns:
         OpenAI客户端实例
     """
    # 获取千问API密钥
    qwen_api_key = os.getenv("QWEN_API_KEY")
    if not qwen_api_key:
        raise ValueError(f"缺少{qwen_api_key}环境变量")

    # 获取千问请求端口URL
    qwen_base_url = os.getenv("QWEN_BASE_URL")

    client = OpenAI(
        # 配置请求密钥:api_key，这里使用千问模型，请用百炼API Key将下行替换。
        # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
        api_key=qwen_api_key,
        base_url=qwen_base_url,
    )
    return client


# 工具定义 - 符合function calling格式
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的天气信息。参数为地点名称，例如：北京、上海",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "地点名称，例如：北京、上海"}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "用于数学计算，输入应为数学表达式，例如'3+5*2'或'(4+6)/2'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，例如：1+2*3、(5+8)/2"}
                },
                "required": ["expression"],
            },
        },
    }
]


# 实现获取天气
def get_weather(location: str) -> str:
    """获取指定地点的天气信息。参数为地点名称，例如：北京、上海"""
    url = "http://weather.cma.cn/api/autocomplete?q=" + urllib.parse.quote(location)
    response = requests.get(url)
    data = response.json()
    if data["code"] != 0:
        return "没找到该位置的信息"
    location_code = ""
    for item in data["data"]:
        str_array = item.split("|")
        if (
                str_array[1] == location
                or str_array[1] + "市" == location
                or str_array[2] == location
        ):
            location_code = str_array[0]
            break
    if location_code == "":
        return "没找到该位置的信息"
    url = f"http://weather.cma.cn/api/now/{location_code}"
    return requests.get(url).text


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        print(f"[执行计算] {expression}")
        # 实际应用中应使用更安全的计算库
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 初始化工具映射
tools = {
    "get_weather": get_weather,
    "calculate": calculate
}


# 工具调用
def invoke_tool(tool_name: str, tool_parameters) -> str:
    """工具调用"""
    if tool_name not in tools:
        return f"函数{tool_name}未定义"
    return tools[tool_name](**tool_parameters)


# React模式系统提示词
def get_system_prompt():
    return "你需要通过思考-行动-观察的循环来回答用户问题。" \
           "可以使用提供的工具获取信息。\n" \
           "思考过程(Thought)：分析问题，决定是否需要调用工具\n" \
           "如果需要调用工具，使用指定的函数调用格式(Action)\n" \
           "获取工具返回结果后(Observation)，继续思考下一步\n" \
           "当获得足够信息后，用'Final Answer: '开头给出最终答案。"


def run(query: str):

    # 系统信息
    system_msg = {"role": "system", "content": get_system_prompt()}

    print(f"\n【系统】:\n{system_msg['content']}")
    print(f"\n【用户】:\n{query}")

    max_iter = 5  # 最大迭代次数
    messages: List[Dict] = [system_msg,
                            {"role": "user", "content": query}]

    # 记录React流程的思考过程
    react_history = ""

    # 获取模型客户端
    client = initialize_model_client()

    for iter_seq in range(1, max_iter + 1):
        print(f"\n\n\n>>> 迭代次数: {iter_seq}")
        print(f">>> 当前React历史:\n{react_history}")

        # 向LLM发起请求，指定可以调用的工具
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="qwen-plus-latest",
            tools=tool_definitions,
            tool_choice="auto",  # 让模型自动决定是否调用工具
            timeout=60
        )

        response = chat_completion.choices[0].message
        content = response.content
        tool_calls = response.tool_calls

        print(f">>> LLM响应内容:\n{content}")
        # 记录思考过程
        react_history += f"Thought: {content}\n"

        # 检查是否有最终答案
        if content and re.search(r"Final Answer:\s*", content, re.DOTALL):
            final_answer = re.sub(r"Final Answer:\s*", "", content, re.DOTALL)
            print(f"\n>>> 最终答案: {final_answer}")
            return

        # 将模型的响应添加到消息历史中
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls
        })

        # 如果没有工具调用，说明模型无法回答或需要更多信息
        if not tool_calls:
            print(">>> 模型没有调用任何工具，也没有给出最终答案")
            react_history += "Observation: 无法确定需要调用的工具\n"
            continue

        # 处理工具调用
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f">>> 调用工具: {function_name}")
            print(f">>> 工具参数: {function_args}")
            # 记录工具调用
            react_history += f"Action: 调用工具 {function_name}，参数 {json.dumps(function_args)}\n"

            # 调用工具并获取结果
            try:
                result = invoke_tool(function_name, function_args)
                print(f">>> 工具返回结果: {result}")
                react_history += f"Observation: {result}\n"

                # 将工具调用结果添加到消息历史中
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })
            except Exception as e:
                error_msg = f"工具调用失败: {str(e)}"
                print(f">>> {error_msg}")
                react_history += f"Observation: {error_msg}\n"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": error_msg
                })
    print("\n>>> 迭代次数达到上限或发生错误，无法生成最终答案")
    print(f">>> 完整React历史:\n{react_history}")


if __name__ == "__main__":
    query = "广州天气如何"
    # query = "计算223344557799*5599"
    run(query)
