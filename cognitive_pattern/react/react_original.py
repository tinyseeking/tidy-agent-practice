import json
import os
import re
from typing import Optional, Dict

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
    # 配置请求密钥:api_key，这里使用千问模型，请用百炼API Key将下行替换。
    # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    qwen_api_key = os.getenv("QWEN_API_KEY")
    if not qwen_api_key:
        raise ValueError(f"缺少环境变量QWEN_API_KEY")

    # 获取千问请求端口URL
    qwen_base_url = os.getenv("QWEN_BASE_URL")

    client = OpenAI(
        api_key=qwen_api_key,
        base_url=qwen_base_url,
    )
    return client


# react 提示词模板
REACT_PROMPT_FORMAT = """
Answer the following questions as best you can. You have access to the following tools:
{tools}


The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{{{
"action": $TOOL_NAME,
"action_input": $INPUT
}}}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


Begin! Reminder to always use the exact characters `Final Answer` when responding. 
"""

# 工具定义
tool_definition = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的天气信息。参数为地点名称，例如：北京、上海",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "location"}
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
                    "expression": {"type": "string", "description": "数学表达式"}
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


# 初始化工具
tools = {
    "get_weather": get_weather,
    "calculate": calculate
}


# 工具调用
def invoke_tool(toolName: str, toolParameters) -> str:
    """工具调用"""
    if toolName not in tools:
        return f"函数{toolName}未定义"
    return tools[toolName](**toolParameters)


# 系统提示词
def get_system_prompt():
    tool_strings = "\n".join([json.dumps(tool["function"], ensure_ascii=False) for tool in tool_definition])
    tool_names = ", ".join([tool["function"]["name"] for tool in tool_definition])
    return REACT_PROMPT_FORMAT.format(tools=tool_strings, tool_names=tool_names)


# 解析提取调用工具及其参数
def parse_action(text: str) -> Optional[Dict[str, str]]:
    """解析模型输出，提取工具调用信息"""
    try:
        # 查找格式为:
        # Action:
        # ```
        # {"action": "...", "action_input": {...}}
        # ```
        action_pattern = re.compile(r"\nAction:\n`{3}(?:json)?\n(.*?)`{3}.*?$", re.DOTALL)
        action_match = action_pattern.search(text)
        if action_match:
            tool_call = json.loads(action_match.group(1))
            return {
                "tool_name": tool_call.get("action"),
                "input": tool_call.get("action_input", {})
            }
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 解析工具调用信息失败 - {str(e)}")
        return None


def run(query: str):
    # 系统信息
    system_msg = {"role": "system", "content": get_system_prompt()}

    print(f"\n【系统】:\n{system_msg['content']}")
    print(f"\n【用户】:\n{query}")

    maxIter = 5  # 最大迭代次数
    agent_scratchpad = ""  # 历史思考、行动、观察记录

    # 获取千问模型客户端
    client = initialize_model_client()

    for iter_seq in range(1, maxIter + 1):
        print(f"\n\n\n>>> 迭代次数: {iter_seq}")

        # 构建消息列表
        messages = [
            system_msg,
            {"role": "user", "content": f"Question: {query}\n\nhistory：\n{agent_scratchpad}"}
        ]

        # 向LLM发起请求
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="qwen-plus-latest",
            # model="gpt-3.5-turbo",  # 使用早期的openai模型测试
            stop="Observation:",
            timeout=30
        )

        content = chat_completion.choices[0].message.content
        if not content:
            print(">>> 错误: LLM返回空内容")
            break

        print(f">>> LLM响应:\n{content}")

        # 检查是否有最终答案，有则直接返回答案
        final_answer_match = re.search(r"Final Answer:\s*(.*)", content, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            print(f"\n>>> 最终答案: {final_answer}")
            return

        # 解析提取工具调用信息
        tool_call = parse_action(content)
        if not tool_call:
            result = "错误: 无法从响应中提取工具调用信息"
            print(f">>> {result}")
            # 更新思考过程
            agent_scratchpad += f"{content}\nObservation: {result}\n"
            continue

        # 执行工具
        try:
            tool_name = tool_call["tool_name"]
            tool_parameters = tool_call["input"]

            print(f">>> 调用工具: {tool_name}")
            print(f">>> 工具参数: {tool_parameters}")

            # 调用工具并获取结果
            result = invoke_tool(tool_name, tool_parameters)
            print(f">>> 工具返回结果: {result}")

            # 更新思考过程
            agent_scratchpad += f"{content}\nObservation: {result}\n"

        except Exception as e:
            print(f">>> 错误: 工具调用失败 - {str(e)}")
            # 更新思考过程
            agent_scratchpad += f"{content}\nObservation: 错误: 工具调用失败 - {str(e)}\n"

    print("\n>>> 迭代次数达到上限或发生错误，无法生成最终答案")


if __name__ == "__main__":
    query = "广州天气如何"
    # query = "计算223344557799*5599"
    run(query)
