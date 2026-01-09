import os
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量,读取.env文件配置信息
load_dotenv()

# 初始化OpenAI客户端（适配千问/OpenAI/DeepSeek/智谱等兼容OpenAI接口规范的大模型）
# 替换说明：更换模型仅需调整 ①API密钥(api_key) ②服务地址(base_url) ③调用时指定的模型名称
client = OpenAI(
    # 千问模型API密钥（必填）：从环境变量读取，避免硬编码泄露
    # 官方文档：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    api_key=os.environ.get("QWEN_API_KEY"),

    # 千问API服务地址（兼容OpenAI格式）：从环境变量读取，适配不同部署环境
    # 默认值参考：https://dashscope.aliyuncs.com/compatible-mode/v1
    base_url=os.environ.get("QWEN_BASE_URL"),
)


class GateKeeper:
    """企业级 AI 问答助手的门控机制实现"""

    def __init__(self, model: str = "qwen-plus"):
        self.model = model

    def safety_gate(self, user_input: str) -> bool:
        """安全门控：检查输入是否安全，不包含有害内容"""
        prompt = f"""
        检查以下用户输入是否安全。安全的定义是：
        1. 不包含暴力、色情、仇恨言论或其他有害内容
        2. 不包含非法活动的请求
        3. 不包含敏感信息的请求

        如果安全，只返回"安全"；如果不安全，只返回"不安全"。

        用户输入：{user_input}
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个安全检查助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip() == "安全"

    def relevance_gate(self, user_input: str, topic: str) -> bool:
        """相关性门控：检查输入是否与指定主题相关"""
        prompt = f"""
        检查以下用户输入是否与主题"{topic}"相关。
        如果相关，只返回"相关"；如果不相关，只返回"不相关"。

        用户输入：{user_input}
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个相关性检查助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip() == "相关"

    def execute_task(self, user_input: str) -> str:
        """执行任务：使用AI模型处理用户请求"""
        system_prompt= """
        你是一个AI助手，擅长完成各种任务。请根据用户的要求提供准确、有用的回答。
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )

        return response.choices[0].message.content

    def quality_gate(self, task_result: str, user_input: str) -> bool:
        """质量门控：评估任务结果的质量"""
        prompt = f"""
        评估以下AI回答是否满足用户的请求。
        一个高质量的回答应该：
        1. 直接回答用户的问题
        2. 提供准确、有用的信息
        3. 内容完整，不缺少关键信息
        4. 语言清晰，易于理解

        如果回答满足以上标准，只返回"高质量"；如果不满足，只返回"低质量"。

        用户请求：{user_input}
        AI回答：{task_result}
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个质量评估助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip() == "高质量"

    def process_request(self, user_input: str, topic: str = "通用") -> Dict[str, Any]:
        """处理用户请求的完整工作流"""
        result = {
            "success": False,
            "user_input": user_input,
            "topic": topic,
            "safety_check": False,
            "relevance_check": False,
            "task_result": None,
            "quality_check": False,
            "final_response": None,
            "reason": None
        }

        # 1. 安全门控
        result["safety_check"] = self.safety_gate(user_input)
        if not result["safety_check"]:
            result["reason"] = "输入内容不安全"
            result["final_response"] = "抱歉，您的请求包含不安全内容，无法处理。"
            return result

        # 2. 相关性门控
        result["relevance_check"] = self.relevance_gate(user_input, topic)
        if not result["relevance_check"]:
            result["reason"] = f"输入与主题'{topic}'不相关"
            result["final_response"] = f"抱歉，您的请求与当前主题'{topic}'不相关，无法处理。"
            return result

        # 3. 执行任务
        result["task_result"] = self.execute_task(user_input)

        # 4. 质量门控
        result["quality_check"] = self.quality_gate(result["task_result"], user_input)
        if not result["quality_check"]:
            result["reason"] = "AI生成的结果质量不达标"
            result["final_response"] = "抱歉，AI生成的回答质量不达标，请尝试重新提问。"
            return result

        # 所有门控都通过
        result["success"] = True
        result["final_response"] = result["task_result"]

        return result


# 使用示例
if __name__ == "__main__":
    # 创建门控代理
    agent = GateKeeper()

    # 测试用例
    test_cases = [
        {
            "input": "请解释一下量子计算的基本原理",
            "topic": "科技"
        },
        {
            "input": "今天天气怎么样",
            "topic": "科技"
        },
        {
            "input": "asdfasdf",  # 无意义输入
            "topic": "科技"
        }
    ]

    # 运行测试用例
    for i, test_case in enumerate(test_cases):
        print(f"\n--- 测试用例 {i + 1} ---")
        print(f"输入: {test_case['input']}")
        print(f"主题: {test_case['topic']}")

        result = agent.process_request(test_case["input"], test_case["topic"])

        print(f"结果: {result['success']}")
        print(f"原因: {result['reason']}")
        print(f"最终响应: {result['final_response']}")
        print("安全检查:", result["safety_check"])
        print("相关性检查:", result["relevance_check"])
        print("质量检查:", result["quality_check"])
