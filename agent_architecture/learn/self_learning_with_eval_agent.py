import json

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
from datetime import datetime

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


def generate_response(llm: ChatOpenAI, history: ChatMessageHistory, insights: str, user_input: str) -> str:
    """生成上下文感知的回复（结合历史+已学洞见）"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一款具备自我优化能力的人工智能助手。从互动交流中学习经验，持续提升自身表现。需满足以下要求：
            1. 回答准确：基于事实，不编造信息；
            2. 详略得当：用户浅问则简答，深问则补充细节；
            3. 贴合需求：优先回应用户核心诉求，再补充相关信息；
            4. 持续改进：结合下方「改进洞见」优化回复风格和内容。
            改进洞见：{insights}
        """),
        MessagesPlaceholder(variable_name="history"),  # 对话历史
        ("human", "{user_input}")
    ])

    # 直接格式化prompt并调用LLM
    formatted_prompt = prompt.format_messages(
        insights=insights,
        history=history.messages,
        user_input=user_input
    )
    response = llm.invoke(formatted_prompt)
    return response.content


def evaluate_response(llm: ChatOpenAI, history: ChatMessageHistory, user_input: str, agent_response: str) -> dict:
    """评估回复质量，输出量化分数和改进方向（1-5分，5分为最优）"""
    # 格式化对话历史（带角色：用户/AI）
    formatted_history = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}：{msg.content}"
        for msg in history.messages
    ])
    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业的AI回复评估师，需从以下4个核心维度全面评估回复质量：
            1. 需求匹配度（relevance）：是否精准捕捉用户核心诉求，无偏离、无遗漏（1-5分）；
            2. 信息准确性（accuracy）：回复内容是否真实可靠，无事实错误或误导性信息（1-5分）；
            3. 详略适配度（detail_level）：是否贴合用户提问深度，既不冗余堆砌，也不遗漏关键信息（1-5分）；
            4. 实用价值度（practicality）：是否能切实帮助用户解决问题、获取有效信息或提供情绪支持（1-5分）。
        
            评估执行要求：
            1. 按上述维度依次打分，每个维度按「优秀（5分）/良好（4分）/合格（3分）/不足（2分）/无效（1分）」打分；
            2. 计算总分（满分20分），保留分数客观性；
            3. 精准指出1-2个最需改进的核心维度，并说明具体原因；
            4. 给出可落地的改进建议（需具体到操作场景，如「用户问‘如何使用工具’时，补充步骤拆解+常见问题规避」）；
            5. 严格按以下格式输出JSON，无额外冗余文字：
            {{"relevance": 需求匹配度分数, "accuracy": 信息准确性分数, "detail_level": 详略适配度分数, "practicality": 实用价值度分数, "total": 总分, "improvement_direction": "需改进的维度及原因", "improvement_suggestion": "具体可落地的改进建议"}}
        """
         ),
        ("human", """对话背景：用户输入是「{user_input}」，AI回复是「{agent_response}」，完整对话历史如下：
        {formatted_history}
        请按要求评估并输出JSON格式结果（仅JSON，无需额外文字）""")
    ])

    formatted_prompt = evaluation_prompt.format_messages(
        user_input=user_input,
        agent_response=agent_response,
        formatted_history=formatted_history
    )
    evaluation = llm.invoke(formatted_prompt)
    # 解析JSON结果（简化异常处理，实际可加重试次数）
    try:
        import json
        return json.loads(evaluation.content)
    except Exception as e:
        return {
            "relevance": 0,
            "accuracy": 0,
            "detail_level": 0,
            "practicality": 0,
            "total": 0,
            "improvement_direction": "评估失败",
            "improvement_suggestion": f"无法解析AI评估结果：{str(e)}"
        }


def reflect_on_evaluation(llm: ChatOpenAI, evaluation: dict, history: ChatMessageHistory) -> str:
    """基于评估结果反思"""
    evaluation_json_str = json.dumps(evaluation, ensure_ascii=False, indent=2)

    # 格式化对话历史（带角色：用户/AI）
    formatted_history = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}：{msg.content}"
        for msg in history.messages
    ])

    # 2. 定义Prompt模板（使用占位符{evaluation}和{history}，避免直接f-string拼接）
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是深度反思专家，需基于评估结果及对话历史进行深度反思：
            1. 先明确评估指出的核心问题（不要重复评估内容，要分析本质）；
            2. 结合对话历史，分析问题产生的原因（如「用户问具体功能时，AI只给了基础介绍，未匹配需求深度」）；
            3. 提出具体的、可操作的改进思路。
            """),
        ("human", """评估结果：{evaluation}
        完整对话历史如下：
        {history}
        请输出简洁的反思结论""")
    ])
    formatted_prompt = reflection_prompt.format_messages(
        evaluation=evaluation_json_str,
        # 对话历史转为字符串（避免列表格式直接传入）
        history=formatted_history
    )
    reflection = llm.invoke(formatted_prompt)
    return reflection.content


def learn_from_reflection(llm: ChatOpenAI, reflection: str, existing_insights: str) -> str:
    """将反思转化为永久的改进洞见（规则化、可复用），合并已有洞见"""
    learning_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是AI的学习系统，需将反思结论转化为「明确、可执行、可复用」的改进洞见及原则：
        1. 原则简洁明确且具体（如「当用户询问产品价格时，需同时告知当前优惠活动和生效时间」）；
        2. 避免空泛表述（不写「提高准确性」，要写「回答产品参数时需核对官方数据，不猜测」）；
        3. 合并已有改进洞见，去重留优（保留最新、最具体的规则）。"""),
        ("human", """
        新反思结论：{reflection}
        已有改进洞见：{existing_insights}
        请输出合并后的改进洞见（分点列出）""")
    ])

    formatted_prompt = learning_prompt.format_messages(
        reflection=reflection,
        existing_insights=existing_insights
    )
    new_insights = llm.invoke(formatted_prompt)
    return new_insights.content


class SelfEvalLearningAgent:
    """自改进智能体：具备会话管理、自我评估、反思与持续改进能力的智能体

    核心特性：
    - 基于大模型实现对话交互
    - 记录会话历史与评估记录，支持改进效果追溯
    - 累计长期改进洞见，实现能力迭代优化
    """

    def __init__(self):

        # 初始化大模型客户端（当前适配千问系列模型）
        # 也可使用其它模型，需要替换：
        # - model: 更换为目标模型标识（如gpt-5、claude-3-opus等）
        # - api_key: 替换为对应平台的API密钥
        # - openai_api_base: 替换为对应平台的API端点（非OpenAI生态需适配）
        self.llm = ChatOpenAI(
            model="qwen-plus-latest",
            api_key=qwen_api_key,
            openai_api_base=qwen_base_url,
            temperature=0.8  # 控制输出随机性：0.8为中等随机性，兼顾创造性与稳定性
        )
        self.session_store = {}  # 存储所有会话的历史：key为会话ID，value为该会话的消息列表
        self.improvement_insights = ""  # 累计的改进洞见（永久保存）
        self.evaluation_records = {}  # 存储评估记录（用于对比改进效果）

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话的对话历史"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    def respond(self, user_input: str, session_id: str) -> str:
        """生成回复并记录对话历史"""
        history = self.get_session_history(session_id)
        response = generate_response(
            llm=self.llm,
            history=history,
            insights=self.improvement_insights,
            user_input=user_input
        )
        # 记录对话
        history.add_user_message(user_input)
        history.add_ai_message(response)
        return response

    def evaluate(self, user_input: str, agent_response: str, session_id: str) -> dict:
        """评估回复质量并记录"""
        history = self.get_session_history(session_id)
        evaluation = evaluate_response(
            llm=self.llm,
            history=history,
            user_input=user_input,
            agent_response=agent_response
        )
        # 记录评估结果（含时间戳）
        if session_id not in self.evaluation_records:
            self.evaluation_records[session_id] = []
        self.evaluation_records[session_id].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_input": user_input,
            "agent_response": agent_response,
            "evaluation": evaluation
        })
        return evaluation

    def reflect(self, evaluation: dict, session_id: str) -> str:
        """基于评估结果定向反思"""
        history = self.get_session_history(session_id)
        return reflect_on_evaluation(llm=self.llm, evaluation=evaluation, history=history)

    def learn(self, reflection: str) -> str:
        """将反思转化为改进洞见"""
        self.improvement_insights = learn_from_reflection(
            llm=self.llm,
            reflection=reflection,
            existing_insights=self.improvement_insights
        )
        # 将新学习的洞见记录到对话历史（供后续参考）
        return self.improvement_insights

    def run_improvement_loop(self, user_input: str, agent_response: str, session_id: str) -> dict:
        """执行完整的改进闭环：评估→反思→学习"""
        print("\n"+"=" * 10 + "开始改进闭环" + "=" * 10 + "\n")
        # 1. 评估
        evaluation = self.evaluate(user_input, agent_response, session_id)
        print(f"评估结果\n：{evaluation}")
        # 2. 反思
        reflection = self.reflect(evaluation, session_id)
        print(f"反思结论\n：{reflection}")
        # 3. 学习
        new_insights = self.learn(reflection)
        print(f"更新后的改进洞见\n：{new_insights}")

        print("\n"+"=" * 10 + "改进闭环结束" + "=" * 10 + "\n")
        return {
            "evaluation": evaluation,
            "reflection": reflection,
            "new_insights": new_insights
        }

    def show_improvement_trend(self, session_id: str) -> None:
        """展示会话的改进趋势（对比评估总分）"""
        records = self.evaluation_records.get(session_id, [])
        if len(records) < 2:
            print("暂无足够评估记录对比改进趋势")
            return
        print("\n=== 改进趋势对比 ===")
        for i, record in enumerate(records):
            print(f"第{i + 1}次互动 - 总分：{record['evaluation']['total']}分 "
                  f"（时间：{record['timestamp']}）")
        print(f"改进幅度：{records[-1]['evaluation']['total'] - records[0]['evaluation']['total']}分")
        print("=== 趋势对比结束 ===\n")


if __name__ == "__main__":
    # 初始化智能体
    agent = SelfEvalLearningAgent()
    session_id = "product_consult_001"
    print("=== 智能手表产品咨询对话 ===")

    # ------------------------------
    # 第一阶段：初始互动（无改进，暴露问题）
    # ------------------------------
    print("\n【第一阶段：初始互动】")
    # 互动1：用户问基础功能（核心需求：了解是否符合日常使用）
    user_input1 = "这款智能手表支持哪些基础功能？"
    response1 = agent.respond(user_input1, session_id)
    print(f"用户：{user_input1}")
    print(f"AI：{response1}")
    # 执行改进闭环（首次评估，发现问题）
    agent.run_improvement_loop(user_input1, response1, session_id)

    # 互动2：用户问价格（核心需求：了解性价比，隐含是否有优惠）
    user_input2 = "它的价格是多少？"
    response2 = agent.respond(user_input2, session_id)
    print(f"\n用户：{user_input2}")
    print(f"AI：{response2}")
    # 执行改进闭环（第二次评估，优化价格相关回复）
    agent.run_improvement_loop(user_input2, response2, session_id)

    # ------------------------------
    # 第二阶段：改进后互动（验证效果）
    # ------------------------------
    print("\n【第二阶段：改进后互动】")
    # 互动3：用户问优惠活动（核心需求：是否有省钱方案）
    user_input3 = "现在买有什么优惠吗？"
    response3 = agent.respond(user_input3, session_id)
    print(f"用户：{user_input3}")
    print(f"AI：{response3}")
    # 执行改进闭环（第三次评估，进一步优化）
    agent.run_improvement_loop(user_input3, response3, session_id)

    # 互动4：用户问使用场景（核心需求：确认是否匹配自身需求）
    user_input4 = "跑步时用它合适吗？有什么特别功能？"
    response4 = agent.respond(user_input4, session_id)
    print(f"\n用户：{user_input4}")
    print(f"AI：{response4}")

    # ------------------------------
    # 展示改进趋势
    # ------------------------------
    agent.show_improvement_trend(session_id)

    # ------------------------------
    # 最终改进洞见总结
    # ------------------------------
    print("\n=== 最终累计改进洞见 ===")
    print(agent.improvement_insights)
