import os
from typing import List

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from tools import TavilySearch

load_dotenv()


def initialize_llm(
        model: str = "qwen-plus",
        temperature: float = 0
) -> BaseChatModel:
    """初始化语言模型(LLM)
    根据配置参数创建并返回一个LLM实例，优先使用环境变量中的配置
    参数:
        model: 模型名称
        temperature: 控制输出随机性的参数，0表示最确定，1表示最随机
    返回:
        初始化好的BaseChatModel实例
    异常:
        ValueError: 当缺少必要的API密钥时抛出
    """
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("缺少环境变量QWEN_API_KEY")

    base_url = os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature
    )


def initialize_tools() -> List:
    """初始化工具列表
       创建并返回智能体可以使用的工具集合，目前包含Tavily搜索工具
    """
    search = TavilySearch(
        max_results=2
    )
    return [search]


# 创建执行器，用来执行计划。
def _create_agent_executor() -> CompiledGraph:
    """创建智能体执行器,使用React模式创建一个能够使用工具执行具体步骤的智能体执行器
    """
    llm = initialize_llm()
    tools = initialize_tools()
    prompt = "你是一个专业的助手。"
    # 创建并返回React模式的智能体
    return create_react_agent(llm, tools, prompt=prompt)


async def main():
    executor = _create_agent_executor()
    # 格式化任务描述，明确告知执行器要完成的工作
    task = f"""2024年澳大利亚网球公开赛男子单打冠军的家乡是哪里？"""
    # 格式化任务描述，明确告知执行器要完成的工作
    task_formatted = f"""\n\n你的任务是执行第1步：{task}。"""

    # 调用执行器执行任务
    agent_response = await executor.ainvoke(
        {"messages": [("user", task_formatted)]}
        , debug=True  # 开启调试，正式应用需关闭
    )
    return agent_response


if __name__ == "__main__":
    import asyncio

    response = asyncio.run(main())
    print(f"response:\n{response['messages'][-1].content}")
