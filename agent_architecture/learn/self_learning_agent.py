from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
# 这里使用阿里千问模型,需配置访问key
# 如何获取千问模型API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
qwen_api_key = os.getenv("QWEN_API_KEY")
if not qwen_api_key:
    raise ValueError("缺少QWEN_API_KEY环境变量")
qwen_base_url = os.getenv("QWEN_BASE_URL")
if not qwen_base_url:  # 默认百炼模型调用服务端口
    qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def generate_response(llm: BaseChatModel, history: ChatMessageHistory, human_input: str, insights: str) -> BaseMessage:
    """生成响应"""
    # 核心Prompt模板,结合用户输入，当前对话历史，以及历史经验和洞见
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "你是一款具备自我优化能力的人工智能助手。从互动交流中学习经验，持续提升自身表现。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        ("system", "近期优化改进洞察： {insights}")
    ])

    # 构建完整的提示上下文
    messages = prompt_template.format_messages(
        history=history.messages,
        input=human_input,
        insights=insights
    )

    # 调用 LLM 生成响应
    response = llm.invoke(messages)

    return response


def reflect_on_response(llm, history: ChatMessageHistory):
    """反思功能"""
    # 构建反思 Prompt.明确反思三要素：核心问题（本质分析）、问题原因（结合历史）、改进思路（可操作）。
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是深度反思专家，需基于对话历史进行深度反思：
            1. 明确指出的回复存在的核心问题（需分析本质）；
            2. 结合对话历史，分析问题产生的原因（如「用户问具体功能时，AI只给了基础介绍，未匹配需求深度」）；
            3. 提出具体的、可操作的改进思路。
            """),
        MessagesPlaceholder(variable_name="history"),
        ("human", """请输出反思结论:""")
    ])

    # 构建消息并调用 LLM
    messages = reflection_prompt.format_messages(history=history.messages)
    reflection_response = llm.invoke(messages)

    return reflection_response.content


def learn_from_reflection(llm, reflection: str, existing_insights: str):
    """学习功能"""
    # 构建学习 Prompt
    learning_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业学习系统，需将反思结论转化为「明确、可执行、可复用」的改进洞见及原则：
        1. 原则简洁明确且具体（如「当用户询问产品价格时，需同时告知当前优惠活动和生效时间」）；
        2. 避免空泛表述（不写「提高准确性」，要写「回答产品参数时需核对官方数据，不猜测」）；
        3. 合并已有改进洞见，去重留优（保留最新、最具体的规则）。"""),
        ("human", """新反思结论：{reflection}
        已有改进洞见：{existing_insights}
        请输出合并后的改进洞见（分点列出）""")
    ])

    # 构建消息并调用 LLM
    messages = learning_prompt.format_messages(
        reflection=reflection,
        existing_insights=existing_insights
    )
    learned_points = llm.invoke(messages).content

    return learned_points


class SelfLearningAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen-plus-latest",  # 模型名称
            api_key=qwen_api_key,
            openai_api_base=qwen_base_url,
            temperature=0.8
        )
        self.store = {}  # 存储会话历史，key: session_id, value: ChatMessageHistory
        self.existing_insights = ""  # 存储反思得到的改进见解

    def get_chat_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话的对话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def respond(self, human_input: str, session_id: str):
        # 获取对话历史
        history = self.get_chat_history(session_id)

        """生成响应并更新对话历史"""
        response = generate_response(
            llm=self.llm,
            history=history,
            human_input=human_input,
            insights=self.existing_insights
        )

        # 将当前交互添加到历史记录
        history.add_user_message(human_input)
        history.add_ai_message(response.content)
        return response.content

    def reflect(self, session_id: str):
        """执行反思并更新见解"""
        # 获取对话历史
        history = self.get_chat_history(session_id)
        return reflect_on_response(self.llm, history)

    def learn(self, session_id: str):
        """先反思再学习，更新知识"""

        # 进行反思
        reflection = self.reflect(session_id)

        # 生成学习要点或洞见
        learn_insights = learn_from_reflection(self.llm, reflection, self.existing_insights)

        # 将学习结果添加到对话历史
        # history = self.get_chat_history(session_id)
        # history.add_ai_message(f"[SYSTEM] Agent learned: {learn_insights}")

        # 积累当前的学习要点和洞见，指导未来的决策。
        self.existing_insights = learn_insights

        return self.existing_insights


if __name__ == "__main__":
    agent = SelfLearningAgent()
    session_id = "user_888"  # 中文场景用户ID示例
    print("=" * 60)
    print("场景测试：成都旅游规划咨询（带自我优化功能）")
    print("=" * 60)

    # 互动1：核心需求咨询（旅游目的地推荐）
    user_input1 = "我计划国庆去成都旅游，推荐3个必去的景点吧"
    print(f"\n我：{user_input1}")
    print(f"AI：{agent.respond(user_input1, session_id)}")

    print("\n\n" + "=" * 60)
    # 互动2：深入需求（景点细节+行程建议）
    user_input2 = "这些景点之间距离远吗？有没有顺路的游玩路线？"
    print(f"\n我：{user_input2}")
    print(f"AI：{agent.respond(user_input2, session_id)}")

    # 学习与改进（Agent自我优化）
    print("\n\n" + "=" * 60)
    print("Agent正在反思对话并学习优化...")
    learned_content = agent.learn(session_id)
    print(f" Agent学到的核心要点：{learned_content}")

    print("\n\n" + "=" * 60)
    # 互动3：美食推荐，验证优化效果
    user_input3 = "在这些景点附近，有哪些本地人常去的川菜馆？不要网红店"
    print(f"\n我：{user_input3}")
    print(f"AI：{agent.respond(user_input3, session_id)}")

    print("\n\n" + "=" * 60)
    # 互动4：实际出行需求，演示持续改进
    user_input4 = "这些餐馆国庆期间需要提前预约吗？有没有人均消费参考？"
    print(f"\n我：{user_input4}")
    print(f"AI：{agent.respond(user_input4, session_id)}")

    print("\n" + "=" * 60)
    print("场景测试完成")
    print(f"改进洞见\n：{agent.existing_insights}")
    print("=" * 60)
