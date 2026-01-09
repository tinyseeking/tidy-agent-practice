import requests
import urllib.parse

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent

from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool

from llm.langchain.langchain_llm import langchain_qwen_llm

# 加载环境变量
load_dotenv()


#######################################
# 本代码展示LangChain 早期用于快速创建ReAct代理的方式,使用initialize_agent 和 AgentExecutor 实现，新版本中已迁移使用LangGraph方式实现。
# initialize_agent 和 AgentExecutor  适合简单的工具调用场景，但存在以下局限：
# 灵活性不足：代理逻辑（如 ReAct 提示）较为固定，难以定制复杂的工作流。
# 状态管理有限：不支持跨会话的状态持久化或错误恢复。
# 工具调用兼容性：对现代聊天模型的工具调用支持不够完善（如 OpenAI 的 gpt-4o）。
# 扩展性差：不适合多步骤、多用户或人机交互的复杂场景。
#######################################

# 实现获取天气的工具函数
def get_weather(location: str) -> str:
    """获取指定地点的天气信息。参数为地点名称，例如：北京、广州"""
    try:
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

    except Exception as e:
        return f"获取天气信息失败：{str(e)}"


# 实现计算工具函数
def calculate(expression: str) -> str:
    """用于数学计算，输入应为数学表达式，例如'3+5*2'或'(4+6)/2'"""
    try:
        # 实际应用中应使用更安全的计算库，如sympy
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def main():
    # 1 定义工具列表
    tools = [
        Tool(
            name="get_weather",
            func=get_weather,
            description="获取指定地点的天气信息。参数为地点名称，例如：北京、上海"
        ),
        Tool(
            name="calculate",
            func=calculate,
            description="用于数学计算，输入应为数学表达式，例如'3+5*2'或'(4+6)/2'"
        )
    ]

    # 2 初始化聊天模型 - 这里使用OpenAI兼容接口，可根据实际情况修改
    llm = langchain_qwen_llm(
        model="qwen-plus-latest",  # 可以替换为其它支持的模型
        temperature=0,
    )

    # 3 初始化记忆（支持多轮对话）
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 4. 初始化ReAct智能体
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # 使用ReAct模式
        verbose=True,  # 打印详细过程
        memory=memory,
        handle_parsing_errors=True  # 处理解析错误
    )

    # 5. 运行示例
    if __name__ == "__main__":
        print("===== 示例1：简单天气查询 =====")
        response = agent.run("北京现在的天气怎么样？需要带伞吗？")
        print("答案:", response)

        print("\n===== 示例2：复杂多步骤任务 =====")
        response = agent.run("""计算123456789*99""")
        print("答案:", response)


if __name__ == "__main__":
    main()
