import os
import operator
from typing import Annotated, List, Tuple, Union

# 导入LangChain相关组件：输出解析器、图结构、类型定义等
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

# 环境变量加载、语言模型接口、提示模板等
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# 导入搜索工具

from tools import TavilySearch


load_dotenv()


# ----------------------------
# 配置与常量
# ----------------------------
class Config:
    """应用配置常量"""

    DEFAULT_MODEL = "qwen-plus-2025-04-28"  # 使用千问模型"qwen-plus" 使用qwen-turbo系列效果差
    DEFAULT_TEMPERATURE = 0  # 默认温度参数（0表示确定性输出）
    TAVILY_MAX_RESULTS = 2  # 搜索工具返回的最大结果数
    QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 千问API基础地址
    RECURSION_LIMIT = 10  # 递归限制，防止无限循环


# ----------------------------
# 状态与数据模型
# ----------------------------
class PlanExecute(TypedDict):
    """智能体运行状态数据结构,用于在工作流的不同节点之间传递状态信息"""
    input: str  # 用户输入的原始查询
    plan: List[str]  # 待执行的计划步骤列表
    # 已执行步骤记录，使用Annotated和operator.add实现状态累加
    past_steps: Annotated[List[Tuple], operator.add]
    response: str  # 最终要返回给用户的响应


class Plan(BaseModel):
    """未来要遵循的计划步骤"""
    steps: List[str] = Field(description="要遵循的不同步骤，应按顺序排列")


class Response(BaseModel):
    """用户响应数据模型"""
    response: str


class Act(BaseModel):
    """要执行的动作,用于在重新规划阶段决定下一步动作：是直接响应还是继续执行计划"""
    action: Union[Response, Plan] = Field(
        description="要执行的动作。若要响应用户，使用Response。若需要进一步使用工具，使用Plan。"
    )


def load_environment():
    """加载环境变量,从.env文件中加载API密钥等敏感信息，避免硬编码"""
    load_dotenv()


# ----------------------------
# LLM初始化
# ----------------------------
def initialize_llm(
        model: str = Config.DEFAULT_MODEL,
        temperature: float = Config.DEFAULT_TEMPERATURE
) -> ChatOpenAI:
    """初始化语言模型(LLM)
    创建并返回一个ChatOpenAI实例，这里默认访问百炼的千问模型，需要从环境变量中读取参数：\n
    QWEN_API_KEY: 访问模型的密钥；\n
    QWEN_BASE_URL: 访问模型的URL接口地址。\n
    支持兼容openai 接口的模型服务，一般只需要替换以上这两个值。

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

    base_url = os.getenv("QWEN_BASE_URL") or Config.QWEN_BASE_URL

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
    # 初始化tavily搜索工具，设置最大结果数
    # 请求网络时，如果是同步则使用requests库发起网络请求，异步使用aiohttp发起网络请求。
    # --同步调用的 requests 库会自动检测系统代理（如环境变量 HTTP_PROXY/HTTPS_PROXY、系统网络设置），
    # --而异步 aiohttp 默认不会。需要自行设置系统代理，实现异步网络请求。
    # 所以这里对langchain默认的TavilySearch做了修改,添加自动获取系统代理并配置.
    search = TavilySearch(max_results=Config.TAVILY_MAX_RESULTS)
    return [search]


# 创建执行器，用来执行计划。
def _create_agent_executor() -> CompiledGraph:
    """创建执行器,使用React模式创建一个能够使用工具执行具体步骤的智能体执行器
    """
    llm = initialize_llm()
    tools = initialize_tools()
    prompt = "你是一个专业的助手。"
    # 创建并返回React模式的智能体
    return create_react_agent(llm, tools, prompt=prompt)


# ----------------------------
# 节点函数
# ----------------------------
class Nodes:
    """智能体图节点集合类

    封装工作流中所有节点的处理函数，使代码结构更清晰
    """

    @staticmethod
    async def plan_step(state: PlanExecute):
        """规划步骤节点：生成初始计划

        根据用户输入生成完成任务所需的步骤计划

        参数:
            state: 当前工作流状态，包含用户输入等信息

        返回:
            包含生成的计划步骤的字典
        """

        # 创建解析器生成格式说明
        parser = PydanticOutputParser(pydantic_object=Plan)

        # 生成初始计划的提示模板，指导模型为目标生成完整的分步计划，确保计划能够直接导向正确答案。
        planner_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """该计划应包含单个任务，若执行正确，将得出正确答案。不要添加任何多余步骤。\
                最后一步的结果应为最终答案。确保每个步骤都包含所需的所有信息——不要跳过步骤。\n
                必须严格按照以下格式要求返回数据，不要添加任何额外内容：\n
                {format_instructions}\n\n
                请确保输出完全符合上述格式，字段类型和约束要严格遵守。
                """,  # 添加了JSON格式要求
            ),
            ("placeholder", "{messages}"),
        ])
        # 创建规划器：提示模板 + LLM（带结构化输出）
        planner = planner_prompt | initialize_llm().with_structured_output(Plan)
        # 调用规划器生成计划
        result = await planner.ainvoke({
            "messages": [("user", state["input"])],
            "format_instructions": parser.get_format_instructions()  # 解析器生成格式说明
        })
        # 返回包含计划步骤，将被合并到工作流状态中
        return {"plan": result.steps}

    @staticmethod
    async def execute_step(state: PlanExecute):
        """执行步骤节点：执行计划中的第一步

        调用智能体执行器处理计划中的第一个步骤，并记录执行结果

        参数:
            state: 当前工作流状态，包含计划步骤等信息

        返回:
            包含已执行步骤记录的字典
        """
        # 1.获取React智能体执行器
        executor = _create_agent_executor()

        # 2.处理计划：格式化步骤列表，提取第一步任务
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]  # 取出计划中的第一个步骤

        # 3.格式化任务描述，明确告知执行器要完成的工作
        task_formatted = f"""对于以下计划：
                            {plan_str}
                            你的任务是执行第1步：{task}。"""

        # 4.调用执行器执行任务
        agent_response = executor.invoke(
            {"messages": [("user", task_formatted)]}
            , debug=True  # 开启调试，正式应用需关闭
        )

        # 5.记录已执行步骤（(任务内容, 执行结果)），返回给LangGraph合并状态
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }

    @staticmethod
    async def replan_step(state: PlanExecute):
        """重新规划步骤节点：根据执行结果更新计划

        分析已执行步骤的结果，决定是继续执行新的步骤还是直接响应用户

        参数:
            state: 当前工作流状态，包含原始计划和已执行步骤

        返回:
            包含新计划或最终响应的字典
        """
        # 1. 定义重新规划的提示模板
        replanner_prompt = ChatPromptTemplate.from_template("""针对给定目标，制定一个简单的分步计划。
        该计划应包含单个任务，若执行正确，将得出正确答案。不要添加任何多余步骤。
        最后一步的结果应为最终答案。确保每个步骤都包含所需的所有信息——不要跳过步骤。

        你的目标是：
        {input}

        你最初的计划是：
        {plan}

        你目前已完成以下步骤：
        {past_steps}

        相应地更新你的计划。如果无需更多步骤且可以回复用户，则进行回复。否则，填写计划。仅在计划中添加仍需完成的步骤。不要将已完成的步骤作为计划的一部分返回。
        
        必须严格按照以下格式要求返回数据，不要添加任何额外内容：\n
        {format_instructions}\n\n
        请确保输出完全符合上述格式，字段类型和约束要严格遵守。
        """)  # 添加了JSON格式要求

        # 2. 创建解析器,用于生成响应格式说明
        parser = PydanticOutputParser(pydantic_object=Act)
        # 3. 创建重新规划器：提示模板 + LLM（带结构化输出）
        replanner = replanner_prompt | initialize_llm().with_structured_output(Act)
        # 4. 调用重新规划器生成新的动作指令
        output = await replanner.ainvoke({
            "input": state.get("input"),
            "plan": state.get("plan"),
            "past_steps": state.get("past_steps"),
            "format_instructions": parser.get_format_instructions()  # 解析器生成格式说明
        })
        # 5. 根据输出的动作类型返回不同的结果
        if isinstance(output.action, Response):
            # 如果是响应动作，返回最终回答
            return {"response": output.action.response}
        else:
            # 如果是计划动作，返回更新后的计划
            return {"plan": output.action.steps}

    @staticmethod
    def should_end(state: PlanExecute) -> str:
        """路由节点：决定是否结束流程

        根据当前状态判断工作流应继续执行还是结束

        参数:
            state: 当前工作流状态

        返回:
            下一步节点名称（"execute"表示继续执行，END表示结束）
        """
        # 若状态中有response（最终答案），则返回END（结束）；否则返回"execute"（继续执行）
        return END if "response" in state and state["response"] else "execute"


# ----------------------------
# 图构建（工作流定义）
# ----------------------------
def build_plan_and_execute_workflow() -> CompiledStateGraph:
    """构建plan-execute工作流图

    定义工作流中的节点和节点之间的连接关系，构建完整的智能体工作流程

    返回:
        编译好的状态图(CompiledStateGraph)
    """
    workflow = StateGraph(PlanExecute)

    # 添加节点
    workflow.add_node("planner", Nodes.plan_step)  # 规划节点
    workflow.add_node("execute", Nodes.execute_step)  # 执行节点
    workflow.add_node("replan", Nodes.replan_step)  # 重新规划节点

    # 定义节点间的连接关系
    workflow.add_edge(START, "planner")  # 起始点 -> 规划节点
    workflow.add_edge("planner", "execute")  # 规划节点 -> 执行节点
    workflow.add_edge("execute", "replan")  # 执行节点 -> 重新规划节点
    # 添加条件边：根据重新规划的结果决定下一步
    workflow.add_conditional_edges(
        "replan",  # 源节点
        Nodes.should_end,  # 路由判断函数
        ["execute", END],  # 可能的目标节点
    )

    # 编译并返回工作流图
    return workflow.compile()


def save_grath_image(app: CompiledStateGraph):
    """保存工作流图为图片

    生成并保存工作流的可视化图表，默认使用mermaid api 生成图片，需要科学上网。

    参数:
        app: 编译好的状态图实例
    """
    from PIL import Image
    import io
    # 获取图像二进制数据（使用mermaid API生成）
    image_bytes = app.get_graph(xray=True).draw_mermaid_png()
    image_stream = io.BytesIO(image_bytes)
    # 保存为本地文件
    with Image.open(image_stream) as img:
        img.save("plan_and_execute.png")  # 保存为PNG格式


# ----------------------------
# 主程序
# ----------------------------
async def main():
    """运行计划-执行智能体示例"""
    # 1. 初始化环境变量
    load_environment()

    # 2. 构建工作流
    app = build_plan_and_execute_workflow()

    # 3. （可选）保存工作流图片（需科学上网）
    # 默认使用mermaid api 生成图片，需要科学上网,才能执行成功
    # save_grath_image(app)

    # 4. 测试查询（用户输入）
    inputs = {"input": "2024年澳大利亚网球公开赛男子单打冠军的家乡是哪里？"}
    # 5. 设置递归限制（防止无限循环）
    config = {"recursion_limit": Config.RECURSION_LIMIT}
    # 6. 流式运行工作流（实时输出每一步结果）
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":  # 过滤掉结束标记
                print(v)


if __name__ == "__main__":
    import asyncio

    # 运行主程序
    asyncio.run(main())
