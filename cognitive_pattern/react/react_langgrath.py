import os

import requests
import urllib.parse
from typing import Annotated, Literal, Optional, Dict, List, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain.tools import tool
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def initialize_qwen_llm(model="qwen-plus", temperature=0):
    """
    初始化千问聊天模型并返回
    """
    # 获取环境中千问 API密钥
    qwen_api_key = os.getenv("QWEN_API_KEY")
    if not qwen_api_key:
        raise ValueError("缺少QWEN_API_KEY环境变量")
    qwen_base_url = os.getenv("QWEN_BASE_URL")
    if not qwen_base_url:
        raise ValueError("缺少qwen_base_url环境变量")

    # 初始化千问聊天模型
    llm = ChatOpenAI(
        model=model,
        api_key=qwen_api_key,
        openai_api_base=qwen_base_url,
        temperature=temperature
    )

    return llm


# 1. 定义工具函数
@tool("get_weather", return_direct=False)
def get_weather(location: str) -> str:
    """获取指定地点的天气信息。参数为地点名称，例如：北京、上海"""
    try:
        # 模拟浏览器的请求头，关键是User-Agent，可根据需要补充其他头
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "http://weather.cma.cn/"
        }

        url = "http://weather.cma.cn/api/autocomplete?q=" + urllib.parse.quote(location)

        # 发送请求时携带请求头
        response = requests.get(url, headers=headers, timeout=60)
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
            return f"未找到【{location}】对应的天气信息"

        weather_url = f"http://weather.cma.cn/api/now/{location_code}"
        # 再次携带请求头请求天气数据
        weather_response = requests.get(weather_url, headers=headers, timeout=10)
        return weather_response.text

    except Exception as e:
        return f"获取天气信息失败：{str(e)}"


@tool("calculate", return_direct=False)
def calculate(expression: str) -> str:
    """计算数学表达式的结果。参数为数学表达式，例如：1+2*3、(5+8)/2"""
    try:
        # 实际应用中应使用更安全的计算库，如sympy
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 工具节点初始化
tools = [get_weather, calculate]


# 2. 定义状态
class State(BaseModel):
    messages: Annotated[List[Any], add_messages] = Field(default_factory=list)

    # 用于跟踪是否需要继续处理
    is_finished: bool = False


# 3. 定义智能体节点
def agent_node(state: State):
    # 初始化LLM
    llm = initialize_qwen_llm(
        model="qwen-plus-latest",  # 可以替换为其它支持的模型
        temperature=0,
    )

    # 绑定工具到LLM
    llm_with_tools = llm.bind_tools(tools)

    # 系统提示
    system_message = {
        "role": "system",
        "content": "你拥有调用工具的能力，可以使用工具来回答用户的问题。"
                   "如果需要调用工具，请使用指定的函数调用格式。"
                   "如果已经获取了足够的信息，可以直接给出最终答案。"
                   "请用中文回答用户的问题。"
    }

    # 构建消息列表
    messages = [system_message] + state.messages

    # 调用LLM
    response = llm_with_tools.invoke(messages)

    # 检查是否有最终答案（没有工具调用）
    if not response.tool_calls:
        return {
            "messages": [response],
            "is_finished": True
        }

    return {
        "messages": [response],
        "is_finished": False
    }


# 4. 定义条件函数 - 决定下一步
def should_continue(state: State) -> Literal["tools", END]:
    if state.is_finished:
        return END
    # 检查最后一条消息是否有工具调用
    last_message = state.messages[-1] if state.messages else None
    if last_message and last_message.tool_calls:
        return "tools"
    return END


# 5. 构建图
def build_graph(visualize: bool = False):
    graph = StateGraph(State)

    # 构建工具节点
    tool_node = ToolNode(tools)

    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # 设置入口点
    graph.set_entry_point("agent")

    # 添加条件边
    graph.add_conditional_edges(
        "agent",
        should_continue,
    )

    # 添加从工具到智能体的边
    graph.add_edge("tools", "agent")

    # 编译图
    compiled_graph = graph.compile()

    # 如果需要可视化，生成 Mermaid 语法格式
    if visualize:
        try:
            # 将图形转换为 Mermaid 语法格式
            # 通过mermaid转换工具将格式转换为图片，如：https://www.min2k.com/tools/mermaid/
            print(compiled_graph.get_graph().draw_mermaid())
        except Exception as e:
            print(f"可视化图结构时出错: {e}")

    return compiled_graph


# 6. 运行示例
def main():
    # 构建图
    app = build_graph(visualize=True, )

    # 测试查询1：天气查询
    query = "广州天气如何"
    print(f"【用户查询】: {query}")
    # # 运行图
    # final_state = app.invoke({"messages": [{"role": "user", "content": query}]})
    # # 输出最终结果
    # print(f"\n【最终答案】:")
    # print(final_state['messages'][-1].content)

    # 流式输出
    for output in app.stream({"messages": [{"role": "user", "content": query}]}):
        for key, value in output.items():
            print(f"节点： {key}")
            print("_______")
            print(value)
            print("\n")


if __name__ == "__main__":
    main()
