import os

from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI, Client

# 加载环境变量
load_dotenv()


# 初始化模型调用客户端
def initialize_model_client() -> Client:
    """
    模型调用客户端,使用openai SDK，支持所有兼容openai接口的模型服务，默认使用千问模型

     Returns:
         OpenAI客户端实例
     """

    # 使用千问模型，需要配置千问模型的访问密钥环境变量：QWEN_API_KEY
    # 千问模型接口访问key
    # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    api_key = os.getenv("QWEN_API_KEY")
    # 获取千问请求端口URL
    api_base = os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    if not api_key:
        raise ValueError("请提供千问API密钥，可通过参数或QWEN_API_KEY环境变量设置")

    # 初始化OpenAI客户端
    return OpenAI(
        api_key=api_key,
        base_url=api_base,
    )


class SelfRefineAgent:
    def __init__(self, llm_client: Client, system_prompt: str = None, evaluate_criteria: list[str] = None,
                 model: str = "gpt-4"):
        """初始化Self-refine实例"""
        self.llm_client = llm_client  # 模型调用客户端
        self.model = model  # 模型名称
        self.system_prompt = system_prompt  # 作为系统提示
        # 存储迭代历史 (输出, 反馈)
        self.history: List[Tuple[str, Dict]] = []
        self.evaluate_criteria = evaluate_criteria or ["准确性：内容是否事实正确",
                                                       "完整性：是否覆盖所有必要信息",
                                                       "清晰度：表达是否简洁易懂",
                                                       "逻辑性：论证是否严密有条理"]  # 评估标准
        self.refine_need_history_record = False  # 在改进回答时，是否添加历史迭代记录，默认是False，开启后token消耗量会快速增加。

    def _generate(self, prompt: str, response_json_format: bool = False, temperature: float = 0.7) -> str:
        """
        调用模型生成内容
            :param prompt: 提示词
            :param response_json_format: 响应格式，是否指定JSON格式输出
            :param temperature : 温度，控制随机性
            :return: 模型响应
        """
        try:
            # print("*" * 50)
            # print(f"请求prompt:\n{prompt}")
            # 包含系统提示的消息列表
            system_prompt = self.system_prompt or "你是一个 helpful 的助手，能够根据用户需求提供准确、详细的回答或解决问题。"
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt },
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
            }

            # 如果指定了响应格式，使用结构化输出
            if response_json_format:
                kwargs["response_format"] = {"type": "json_object"}
                response = self.llm_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            response = self.llm_client.chat.completions.create(
                **kwargs
            )
            result = response.choices[0].message.content
            return result
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            raise

    def initial_generation(self, input_text: str) -> str:
        """生成初始输出"""
        prompt = f"""
        请完成以下任务：
        任务：{input_text}
        请提供完整、准确、清晰的回答，确保信息全面且有逻辑性。
        """
        initial_output = self._generate(prompt)
        self.history = [(initial_output, {})]  # 初始化历史记录
        return initial_output

    def generate_feedback(self, input_text: str, current_output: str) -> Dict:
        """生成结构化反馈
        Returns:
            结构化反馈字典:
            - needs_improvement: 布尔值，表示是否需要进一步优化
            - suggestions: 数组，包含需要优化的地方或具体建议
        """

        # 构建反馈提示词内容
        prompt_parts = [
            "请仔细评估当前回答是否全面、准确地回应用户输入的核心需求与细节。若存在改进空间，请提供具体、可操作的优化建议。",
            f"用户输入：{input_text}",
            "当前回答：",
            f"{current_output}\n"
        ]

        # 添加反馈标准（引导模型给出结构化建议）
        if self.evaluate_criteria:  # 检查是否有实际内容（非空且非仅空白字符）
            criteria = "\n".join([f"- {c}" for c in self.evaluate_criteria])
            prompt_parts.extend([
                "评估标准如下：",
                f"{criteria}\n"
            ])

        # 添加以JSON格式输出要求
        prompt_parts.extend([
            "请按照以下结构提供JSON格式的评估结果：",
            "{",
            '  "needs_improvement": 布尔值,  // 表示是否需要进一步优化。True:表示需要优化，并写出优化建议; False:表示不需要优化',
            '  "suggestions": ["改进说明1", "改进说明2", ...], // 数组，包含需要优化的地方或具体建议（每项为字符串）',
            '  "global_evaluate": "整体评价" // 如果不需要优化，写出你的理由',
            "}",
            "\n评估结果（仅返回JSON，不要添加其他内容）："
        ])

        # 合并所有部分形成最终反馈prompt
        prompt = "\n".join(prompt_parts)

        feedback_str = self._generate(prompt, temperature=0.7)

        # 解析JSON反馈为字典
        try:
            import json
            feedback = json.loads(feedback_str)
            # 确保返回结构完整
            if 'needs_improvement' not in feedback:
                feedback['needs_improvement'] = False
            if 'suggestions' not in feedback:
                feedback['suggestions'] = []
            return feedback
        except Exception as e:
            print(f"解析反馈JSON出错: {str(e)}")
            # 返回默认反馈结构
            return {
                'needs_improvement': True,
                'suggestions': ["反馈解析错误，请重新优化"]
            }

    def refine_output(self, input_text: str, current_output: str, feedback: Dict) -> str:
        """基于反馈优化输出"""
        # 构建历史上下文
        history_context = ""
        if self.refine_need_history_record:  # 添加历史迭代记录
            history_context = "\n".join([
                f"第{i + 1}次输出: {output}"
                for i, (output, fb) in enumerate(self.history)
            ])

        # 优化提示词
        prompt = f"""
        请根据以下建议，改进你的回答。如有可同时参考历史迭代记录。
        改进后的回答应更准确、全面、详细，并且逻辑清晰。
        不要简单重复之前的内容，确保解决了所有提出的问题。
        用户输入: {input_text}

        你当前的回答:
        {current_output}

        对回答的改进建议:
        {feedback.get('suggestions', [])}

        历史迭代记录: {history_context}
        """
        refined_output = self._generate(prompt)
        return refined_output

    def run(self, input_text: str, max_iterations: int = 3) -> Dict[str, Any]:
        """运行完整的Self-refine(生成→评估→优化)流程”的完整迭代流程 """
        print(f"初始输入: {input_text}\n")

        # 生成初始回复
        current_output = self.initial_generation(input_text)
        print(f"第1次生成: {current_output}\n")

        for i in range(max_iterations):
            # 生成反馈
            feedback = self.generate_feedback(input_text, current_output)
            print("*" * 50)
            print(f"第{i + 1}次反馈:")
            print(f"  是否需要优化: {feedback['needs_improvement']}")
            print(f"  优化建议: {feedback['suggestions']}")
            print(f"  整体评价: {feedback['global_evaluate']}\n")

            # 检查是否需要停止迭代
            if not feedback['needs_improvement']:
                print("迭代终止：模型认为无需进一步优化")
                break

            # 优化输出
            current_output = self.refine_output(input_text, current_output, feedback)
            self.history.append((current_output, feedback))
            print(f"第{i + 2}次生成: {current_output}\n")

        return {
            "final_output": current_output,
            "history": self.history,
            "iterations": len(self.history)
        }


# 示例使用
if __name__ == "__main__":
    # 初始化模型调用客户端
    client = initialize_model_client()

    print("-" * 50)
    # 问题
    user_prompt = "讲一个关于童年的脱口秀"
    # 初始化
    refiner1 = SelfRefineAgent(
        llm_client=client,
        model="qwen-plus-latest",
        # evaluate_criteria=[] # 添加自定义评估标准
    )
    # 运行对话
    final_result1 = refiner1.run(
        input_text=user_prompt,
        max_iterations=3
    )

    # 输出结果
    print("\n--- 最终结果 ---")
    print(f"\n迭代次数：{final_result1['iterations']}")
    print(f"最终结果: {final_result1['final_output']}")

    # # 问题2
    # print("-" * 50)
    # user_prompt = "解释什么是机器学习，包括其主要类型、核心原理和实际应用场景，并比较它与传统编程的区别"
    # refiner2 = SelfRefineAgent(
    #     llm_client=client,
    #     model="qwen-plus-latest",
    #     system_prompt="生成友好、专业且有帮助的对话回复",
    #     evaluate_criteria=[   # 自定义评估标准
    #         "定义准确性：是否准确描述概念本质",
    #         "内容完整性：是否涵盖关键分类和特点",
    #         "示例相关性：举例是否恰当且有代表性",
    #         "技术深度：是否包含足够的技术细节"
    #     ]
    # )
    # # 运行对话
    # final_result2 = refiner2.run(
    #     input_text=user_prompt,
    #     max_iterations=3
    # )
    # # 输出结果
    # print("\n--- 最终结果 ---")
    # print(f"\n迭代次数：{final_result2['iterations']}")
    # print(f"最终结果: {final_result2['final_output']}")
