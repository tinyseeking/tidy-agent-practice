import json
from typing import Dict, List, Any

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


def generate_response(llm: ChatOpenAI, user_query: str, insights: str,
                      history: ChatMessageHistory) -> str:
    """
     生成上下文感知的AI回复，核心是结合对话历史和已积累的优化洞见，提升回复质量

    参数说明：
        llm: ChatOpenAI实例
            用于生成回复的OpenAI聊天模型客户端，已完成初始化配置（如API密钥、模型版本等）
        user_query: str
            用户当前输入的查询文本，即需要回应的即时需求
        insights: str
            已积累的回复优化洞见，用于约束和提升回复质量
        history: ChatMessageHistory
            对话历史记录对象，包含过往的用户消息和AI回复，支持上下文连贯

    返回值：
        str: LLM生成的最终回复内容，已结合上下文和优化洞见
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个持续改进的通用AI助手，需遵循以下通用改进原则优化回复：
            {insights}
            回复要求：
            1. 精准匹配用户核心需求，不偏离主题；
            2. 提供实用、清晰、完整的信息/方案/情绪价值；
            3. 预判用户潜在疑问，主动补充关键细节；
            4. 逻辑连贯，表述简洁易懂。"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}")
    ])

    # 直接格式化prompt并调用LLM
    formatted_prompt = prompt.format_messages(
        history=history.messages,
        insights=insights,
        user_input=user_query
    )
    return llm.invoke(formatted_prompt).content


def evaluate_response(llm: ChatOpenAI, user_query: str, agent_response: str,
                      history: ChatMessageHistory) -> Dict[str, Any]:
    """评估回复质量，无论场景，均从4个核心维度评估回复价值，输出量化分数和改进方向（1-5分，5分为最优）"""

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
        ("human", """对话背景：用户输入是「{user_query}」，AI回复是「{agent_response}」，完整对话历史如下：
        {formatted_history}
        请按要求评估并输出JSON格式结果（仅JSON，无需额外文字）""")
    ])

    formatted_prompt = evaluation_prompt.format_messages(
        user_query=user_query,
        agent_response=agent_response,
        formatted_history=formatted_history
    )
    eval_str = llm.invoke(formatted_prompt).content.strip()
    # 解析JSON结果（简化异常处理，实际可加重试次数）
    try:
        import json
        return json.loads(eval_str)
    except Exception as e:
        return {
            "accuracy": 0,
            "relevance": 0,
            "detail_level": 0,
            "Practicality": 0,
            "total": 0,
            "improvement_direction": "评估失败",
            "improvement_suggestion": f"无法解析AI评估结果：{str(e)}"
        }


def reflect_on_evaluation(llm: ChatOpenAI, evaluation: Dict[str, Any], history: ChatMessageHistory) -> str:
    """基于评估结果和完整对话历史，进行深度反思并输出改进思路

    参数说明：
        llm: ChatOpenAI实例
            用于执行反思逻辑的OpenAI聊天模型客户端（已完成初始化配置）
        evaluation: Dict[str, Any]
            对AI历史回复的评估结果字典，包含问题指出、评分等关键信息（结构灵活，需通过JSON序列化传递）
        history: ChatMessageHistory
            完整的对话历史记录对象，包含过往所有用户消息和AI回复，用于追溯问题上下文

    返回值：
        str: 简洁的反思结论，包含核心问题本质、产生原因及可操作的改进思路
    """
    # 将评估结果字典序列化为JSON字符串，确保特殊字符不转义且格式美观，便于LLM读取
    evaluation_json_str = json.dumps(evaluation, ensure_ascii=False, indent=2)

    # 格式化对话历史：按「角色：内容」的格式拼接，让LLM清晰区分用户与AI的交互流程
    formatted_history = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}：{msg.content}"
        for msg in history.messages
    ])

    # 构建反思提示模板：明确LLM的反思专家角色
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


def learn_from_reflection(llm: ChatOpenAI, reflection: str, existing_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将反思转化为永久的改进洞见（规则化、可复用），合并已有洞见"""

    formatted_existing_insights = "\n".join([
        f"原则：{principle['principle']}  关键词：{', '.join(principle['keywords'])}  适用场景：{principle['applicable_scenarios']}  价值：{principle['value']}   沉淀时间：{principle['timestamp']}"
        for principle in existing_insights
    ])

    learning_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是通用学习系统，需将反思结论转化为「明确、可执行、跨场景可复用」的改进洞见或原则.
        输出格式为一个JSON对象，包含以下字段：
            - "principle": (字符串) 简洁明确且具体的通用原则或洞见，避免空泛表述，多个用分号分割；
            - "keywords": (列表) 3-5个便于后续检索匹配的关键词；
            - "applicable_scenarios": (字符串) 原则的适用场景；
            - "value": (字符串) 遵循该原则的价值。
        要求：
            1. 分析新的反思结论，生成改进洞见。
            2. 如果存在同类场景的洞见，则将新生成的洞见与提供的已有同类场景洞见进行合并，仅保留最新、最具体、最优的一个。
            
        你的最终输出必须是一个纯净的JSON对象，不要包含任何其他解释、文字或代码块标记。
        ## 输出JSON格式：
          {{
            "principle": "在回答编程问题时，应同时提供代码和详细解释",
            "keywords": ["编程", "代码解释"],
            "applicable_scenarios": "适用于用户询问具体编程实现时",
            "value": "帮助用户理解逻辑，减少后续提问"
          }}
        """
         ),
        ("human", """新反思结论：{reflection},
        已有改进洞见：{existing_insights}。
        请输出合并后的改进洞见，请按要求输出JSON""")
    ])

    formatted_prompt = learning_prompt.format_messages(
        reflection=reflection,
        # 已有改进洞见
        existing_insights=formatted_existing_insights
    )
    learn_str = llm.invoke(formatted_prompt).content.strip()
    # 异常处理
    try:
        import json
        return json.loads(learn_str)
    except Exception as e:
        return {
            "error": "An unexpected error occurred during processing.",
            "raw_reflection": reflection,
            "raw_llm_output": learn_str,
            "error_details": str(e)
        }


class LearningMemory:
    """
    对话记忆与长期经验积累核心类（生产环境建议持久化到外部知识库：如Redis/MySQL/向量数据库）
    核心功能：
    1. 会话级记忆（chat_history）：维护单一会话的上下文连贯性，确保LLM记得当前对话的前后内容
    2. 长期经验记忆（improvement_insights）：沉淀跨会话、可复用的改进洞见，支持关键词快速召回
       最终实现「系统越用越聪明」的自学习效果
    """

    def __init__(self):
        """初始化记忆存储结构：会话历史字典 + 长期改进洞见列表"""

        # 会话级历史对话记录
        self.chat_history: Dict[str, ChatMessageHistory] = {}
        # 长期改进洞见存储：列表形式存储所有跨场景经验，每条洞见包含场景、关键词、改进规则等
        # 洞见结构示例：
        # {
        #     "principle": "在回答编程问题时，应同时提供代码和详细解释",
        #     "keywords": ["编程", "代码解释"],
        #     "applicable_scenarios": "适用于用户询问具体编程实现时",
        #     "value": "帮助用户理解逻辑，减少后续提问"
        # }
        self.improvement_insights: List[Dict[str, Any]] = []

    def get_chat_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话历史记忆（存储当前对话上下文）"""
        if session_id not in self.chat_history:
            self.chat_history[session_id] = ChatMessageHistory()
        return self.chat_history[session_id]

    def add_improvement_insights(self, learning_principle: Dict[str, Any]) -> None:
        """
            新增或更新长期改进洞见（支持幂等性，避免重复存储相同场景的洞见）
            - 如果 场景applicable_scenarios 已存在，则更新对应条目。
            - 如果 applicable_scenarios 不存在，则添加新条目。
        """
        # 确保新条目有时间戳（更新时也刷新时间戳）
        learning_principle["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 查找是否存在具有相同 applicable_scenarios 的条目
        found_index = -1
        for i, item in enumerate(self.improvement_insights):
            if item["applicable_scenarios"] == learning_principle["applicable_scenarios"]:
                found_index = i
                break

        # 根据查找结果进行更新或添加
        if found_index != -1:
            # 找到了，进行更新
            self.improvement_insights[found_index].update(learning_principle)
        else:
            # 没找到，进行添加
            self.improvement_insights.append(learning_principle)

    def get_relevant_improvement_insights(self, user_query: str) -> List[Dict[str, Any]]:
        """根据用户当前查询，检索最相关的长期改进洞见（支持跨场景复用）"""
        # 简单匹配：基于查询关键词与原则或洞见的相关性（可优化为向量检索）
        relevant = []
        for principle in self.improvement_insights:
            if any(keyword in user_query.lower() for keyword in principle["keywords"]):
                relevant.append(principle)
        # 若无直接相关，返回后10条通用原则
        return relevant if relevant else self.improvement_insights[-10:]


class SelfEvalLearningAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="qwen-plus-latest",  # 模型名称
            api_key=qwen_api_key,
            openai_api_base=qwen_base_url,
            temperature=0.8
        )
        self.evaluation_records = {}  # 存储评估记录（用于对比改进效果）
        self.memory = LearningMemory()  # 通用记忆库,存储所有会话的历史、累计的改进洞见（永久保存）

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """格式化对话历史（便于评估和回复生成）"""
        history = self.memory.get_chat_history(session_id)
        return history

    def respond(self, user_query: str, session_id: str) -> str:
        """生成回复并记录对话历史"""
        history = self.get_session_history(session_id)

        # 格式化已累积的洞见
        improvement_insights = self.memory.get_relevant_improvement_insights(user_query)
        principles_format = "\n".join([
            f"- 原则：{p['principle']}（适用场景：{p['applicable_scenarios']}）"
            for p in improvement_insights
        ]) if improvement_insights else "暂无改进洞见"

        # 调用生成响应
        response = generate_response(
            llm=self.llm,
            user_query=user_query,
            history=history,
            insights=principles_format
        )
        # 记录对话
        history.add_user_message(user_query)
        history.add_ai_message(response)
        return response

    def evaluate(self, user_query: str, agent_response: str, session_id: str) -> dict:
        """评估回复质量并记录"""
        history = self.get_session_history(session_id)

        # 评估响应
        evaluation = evaluate_response(
            llm=self.llm,
            history=history,
            user_query=user_query,
            agent_response=agent_response
        )
        # 记录评估结果（含时间戳）
        if session_id not in self.evaluation_records:
            self.evaluation_records[session_id] = []
        self.evaluation_records[session_id].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": user_query,
            "agent_response": agent_response,
            "evaluation": evaluation
        })
        return evaluation

    def reflect(self, evaluation: dict, session_id: str) -> str:
        """基于评估结果定向反思"""
        history = self.get_session_history(session_id)

        # 基于评估和历史交互生成反思结论
        return reflect_on_evaluation(llm=self.llm, evaluation=evaluation, history=history)

    def learn(self, reflection: str, user_query: str) -> Dict[str, Any]:
        """将反思转化为改进洞见"""
        improvement_insights = learn_from_reflection(
            llm=self.llm,
            reflection=reflection,
            existing_insights=self.memory.get_relevant_improvement_insights(user_query)
        )
        return improvement_insights

    def run_improvement_loop(self, user_query: str, agent_response: str, session_id: str) -> dict:
        """执行完整的改进学习闭环：评估→反思→学习"""
        print("=" * 30 + "开始改进闭环" + "=" * 30 + "\n")
        # 1. 评估
        evaluation = self.evaluate(user_query, agent_response, session_id)
        print("\n" + "*" * 10 + "评估结果" + "*" * 10 + "\n")
        print(f"{evaluation}")
        # 2. 反思
        reflection = self.reflect(evaluation, session_id)
        print("\n" + "*" * 10 + "反思结论" + "*" * 10 + "\n")
        print(f"{reflection}")
        # 3. 学习
        new_insights = self.learn(reflection, user_query)
        print("\n" + "*" * 10 + "改进洞见" + "*" * 10 + "\n")
        print(f"{new_insights}")

        print("=" * 30 + "改进闭环结束" + "=" * 30 + "\n")

        # 4. 更新学习洞见到记忆中，若成功生成洞见记录，存入记忆中保存
        if "error" not in new_insights:
            self.memory.add_improvement_insights(new_insights)

        return {
            "evaluation": evaluation,
            "reflection": reflection,
            "new_insights": new_insights
        }

    def show_improvement_insights(self) -> None:
        """展示所有沉淀的通用改进原则"""
        print("\n" + "=" * 70)
        print("【已学习的所有改进洞见】")
        print("=" * 70)
        if not self.memory.improvement_insights:
            print("暂无沉淀的通用原则")
            return
        for idx, principle in enumerate(self.memory.improvement_insights, 1):
            print(f"\n{idx}. 原则：{principle['principle']}")
            print(f"   关键词：{', '.join(principle['keywords'])}")
            print(f"   适用场景：{principle['applicable_scenarios']}")
            print(f"   价值：{principle['value']}")
            print(f"   沉淀时间：{principle['timestamp']}")

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
    # 执行改进闭环（第四次评估，进一步优化）
    agent.run_improvement_loop(user_input3, response3, session_id)
    # ------------------------------
    # 展示改进趋势
    # ------------------------------
    agent.show_improvement_trend(session_id)

    # ------------------------------
    # 展示最终沉淀的原则洞见
    # ------------------------------
    agent.show_improvement_insights()
