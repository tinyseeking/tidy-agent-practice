from typing import List, Iterable, Dict, Any

from datetime import datetime
import json
import requests
from openai.types.chat import ChatCompletionMessageParam

from llm.call_llm import init_model_client
from tools.weather.weather_hefeng import get_current_weather, search_city_info, get_forecast_weather


#####################################
# 工具函数实现
#####################################

def get_current_request_ip_and_geoinfo() -> str:
    """
    获取当前请求的IP地址及对应的地理位置信息
    包括：国家、地区、城市、经纬度等

    Returns:
        str: 格式化后的地理信息JSON字符串
    """
    url = "http://ip-api.com/json/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, ensure_ascii=False)
    else:
        return json.dumps({"error": "Unable to fetch request ip and city data"})


def get_geo_info_by_ip(ip: str) -> str:
    """
    查询指定IP地址对应的地理位置信息,如国家、地区、城市、经纬度等

    Args:
        ip (Optional[str]): 要查询的IP地址

    Returns:
        str: 格式化后的地理信息JSON字符串（含中文）
    """
    if ip is None:
        return json.dumps({"error": "未提供查询的IP地址"})
    url = f"http://ip-api.com/json/{ip}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, ensure_ascii=False)
    else:
        return json.dumps({"error": "Unable to fetch request ip and geo data"})


def get_current_time() -> str:
    """
    获取当前的日期和时间（格式化输出）

    Returns:
        str: 格式化的当前时间字符串
    """
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前日期和时间：{formatted_time}。"


##########################################
# 定义工具描述清单，严格遵循 OpenAI Tools 格式
##########################################

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_request_ip_and_geoinfo",
            "description": "获取当前请求的IP地址及对应的地理位置信息,如国家、地区、城市、经纬度等",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_geo_info_by_ip",
            "description": "查询指定IP地址对应的地理位置信息,如国家、地区、城市、经纬度等",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {
                        "type": "string",
                        "description": "IP地址"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_city_info",
            "description": "根据关键词获取城市基本信息，包括城市的Location ID，多语言名称、经纬度、时区、海拔、Rank值、所在行政区域等",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词（城市名称、拼音等）"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的实时天气情况，包括实时温度、体感温度、风力风向、相对湿度、大气压强、降水量、能见度等",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_id": {
                        "type": "string",
                        "description": "城市Location ID（可通过search_city获取）或以英文逗号分隔的经度,纬度坐标，例如 location=101010100 或 "
                                       "location=116.41,39.92"
                    }
                },
                "required": ["location_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast_weather",
            "description": "获取指定城市的未来几天天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_id": {
                        "type": "string",
                        "description": "城市Location ID（可通过search_city获取）或以英文逗号分隔的经度,纬度坐标，例如 location=101010100 或 "
                                       "location=116.41,39.92"
                    },
                    "days": {
                        "type": "int",
                        "enum": [3, 7, 10, 15, 30],
                        "description": "预报天数（支持最多30天预报，默认7天），可选值：3、7、10、15、30"
                    }
                },
                "required": ["location_id"]
            }
        }
    },
]

# 工具注册表：映射工具名称到具体实现函数
TOOL_REGISTRY: Dict[str, Any] = {
    "get_current_request_ip_and_geoinfo": get_current_request_ip_and_geoinfo,
    "get_geo_info_by_ip": get_geo_info_by_ip,
    "get_current_time": get_current_time,
    "search_city_info": search_city_info,
    "get_current_weather": get_current_weather,
    "get_forecast_weather": get_forecast_weather,
}

# ======================== 模型客户端与响应处理 ========================

# 初始化模型客户端,默认使用千问模型,支持 "qwen", "openai", "deepseek","zhipu"
client = init_model_client("qwen")


# 封装模型响应函数
def get_response(
        messages: Iterable[ChatCompletionMessageParam],
        tools: List[Dict[str, Any]]
):
    """
    封装模型调用逻辑，获取模型响应

    Args:
        messages: 对话消息列表
        tools: 工具定义列表

    Returns:
        Any: 模型响应对象
    """
    response = client.chat.completions.create(
        model='qwen-plus-latest',  # 千问最新模型的plus版本,模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        temperature=0,
        tools=tools
    )
    return response


# ======================== 对话执行逻辑 ========================
def run_conversation(messages: List[Dict[str, Any]]):
    """
    执行对话流程，处理模型的工具调用和响应生成

    Args:
        messages: 初始对话消息列表（包含系统提示和用户问题）
    """
    # 控制最大的执行工具调用次数
    max_function_calls = 5
    function_call_count = 1

    while function_call_count < max_function_calls:
        # 获取模型响应
        first_response = get_response(messages, tool_definitions)
        assistant_output = first_response.choices[0].message
        print(f"\n第{function_call_count}轮，输出信息：{assistant_output}\n")

        # 将模型输出添加到消息列表
        messages.append(assistant_output)

        # 检查是否需要调用工具
        if not assistant_output.tool_calls:
            # 如果模型判断无需调用工具，则将 assistant 的回复直接打印出来，无需进行模型的第二轮调用
            print(f"最终答案：{assistant_output.content}")
            return

        # 处理工具调用（当前仅处理第一个工具调用）
        tool_call = assistant_output.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        # 验证工具是否存在
        if function_name not in TOOL_REGISTRY:
            error_msg = f"工具{function_name}未注册"
            print(f"\n【第{function_call_count}轮】工具调用失败：{error_msg}")
            messages.append({
                "role": "tool",
                "name": function_name,
                "tool_call_id": tool_call.id,
                "content": json.dumps({"error": error_msg}, ensure_ascii=False)
            })
            continue

        # 执行工具调用（工具已注册）
        function_to_call = TOOL_REGISTRY[function_name]
        print(f"\n【第{function_call_count}轮】执行工具调用：{function_name}，参数：{function_args}")

        try:
            result = function_to_call(**function_args)
            print(f"【第{function_call_count}轮】工具调用结果：{result}")
        except Exception as e:
            # 捕获工具执行异常
            result = json.dumps({"error": f"工具执行失败：{str(e)}"}, ensure_ascii=False)
            print(f"【第{function_call_count}轮】工具执行异常：{str(e)}")

        # 添加工具执行结果
        messages.append({
            "role": "tool",
            "name": function_name,
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False)
        })

        function_call_count += 1  # 执行轮次加1

    # 超过最大调用次数提示
    if function_call_count >= max_function_calls:
        print(f"\n【提示】已达到最大工具调用次数（{max_function_calls}次），对话结束")


if __name__ == '__main__':

    """运行天气助手对话交互"""
    print("欢迎使用天气助手！可查询天气、时间信息，输入'exit'可退出程序。")

    while True:
        # 获取用户输入
        prompt = input("\n请输入您的问题：")

        # 检查是否退出
        if prompt.strip().lower() == "exit":
            print("感谢使用，再见！")
            break

        # 若输入为空，提示重新输入
        if not prompt.strip():
            print("输入不能为空，请重新输入。")
            continue

        # 将用户输入添加到消息列表
        messages = [
            # 这个系统消息很重要，如何不加这个，在回答如“福州明天的天气如何”时不能正确回复。# 这个很重要，如何不加这个，在回答如“福州明天的天气如何”时不能正确回复。
            {"role": "system", "content": "你是天气助手，可以使用工具来回答用户的问题，当调用了一个工具无法回答时，检查下选择的工具及器参数是否"
                                          "正确，必要时调整后重新尝试，尽量准确回答用户的问题"},
            {"role": "user", "content": prompt}
        ]

        # 此处可添加处理逻辑（如调用天气查询函数）
        # 示例：假设处理后得到回复
        run_conversation(messages)
