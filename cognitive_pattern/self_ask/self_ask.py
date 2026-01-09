import os
from typing import Optional, List, Dict

from dotenv import load_dotenv
from openai import OpenAI, Client
from tavily import TavilyClient

# 加载环境变量
load_dotenv()


def initialize_model_client() -> Client:
    """
     获取模型调用客户端,使用openai SDK，支持所有兼容openai接口的模型服务，默认使用千问模型

     Returns:
         OpenAI客户端实例
     """
    # 获取千问API密钥
    # 千问模型接口访问key
    # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    qwen_api_key = os.getenv("QWEN_API_KEY")  # 从环境变量获取
    if not qwen_api_key:
        raise ValueError(f"缺少环境变量QWEN_API_KEY")

    # 获取千问请求端口URL
    qwen_base_url = os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    client = OpenAI(
        api_key=qwen_api_key,
        base_url=qwen_base_url,
    )
    return client


class SelfAskAgent:
    def __init__(self, client: Client, llm_model: str = "qwen-plus-latest", max_steps: int = 5):
        """
        初始化Self-Ask代理
        :param client: 模型调用客户端
        :param llm_model: 用于生成子问题和答案的LLM模型,默认使用千问模式:qwen-plus-latest
        :param max_steps: 最大推理步骤（防止无限循环）
        """
        self.client = client
        self.llm_model = llm_model
        self.max_steps = max_steps
        self.history: List[Dict] = []  # 记录推理历史：子问题、中间答案

    def _call_llm(self, prompt: str) -> str:
        """调用LLM生成回答"""
        try:

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0  # 降低随机性，保证输出稳定
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return ""

    def _search(self, query: str) -> Optional[str]:
        """使用搜索答案"""
        # tavily 搜索API KEY
        tavily_api_Key = os.getenv("TAVILY_API_KEY")

        if not tavily_api_Key:
            print("未配置tavily_api_Key，跳过搜索")
            return None

        client = TavilyClient(api_key=tavily_api_Key)
        try:
            # 调用Tavily API执行搜索
            response = client.search(query, max_results=2)

            # 提取搜索结果项
            items = response.get("results", [])
            # 提取所有摘要并合并为字符串（用换行分隔）
            combined_summary = "\n".join([item["content"] for item in items])
            print(f"搜索结果：\n{combined_summary}")
            return combined_summary

        except Exception as e:
            print(f"搜索过程中发生错误: {e}")
            return None

    def _get_followup_prompt(self, query: str, history: str) -> str:
        """生成用于判断是否需要子问题的prompt"""
        return f"""
        问题：{query}
        已有的推理过程：{history}
        接下来还需要问什么问题吗？请仅回答"Yes"或"No"。如果回答Yes，请同时给出具体的追问（格式：Follow up:[你的问题]）。
        """

    def _get_final_answer_prompt(self, query: str, history: str) -> str:
        """生成用于综合中间答案得到最终结果的prompt"""
        return f"""
        请根据以下问题和推理过程，给出最终答案：
        问题：{query}
        推理过程：{history}
        请直接给出答案，无需额外解释。
        """

    def run(self, query: str) -> str:
        """运行Self-Ask流程"""
        self.history = []
        current_history = ""
        step = 0

        while step < self.max_steps:
            step += 1
            # 1. 判断是否需要继续提问
            followup_prompt = self._get_followup_prompt(query, current_history)
            followup_response = self._call_llm(followup_prompt)

            # 2. 解析是否需要子问题
            if "No" in followup_response:
                # 无需更多子问题，生成最终答案
                final_prompt = self._get_final_answer_prompt(query, current_history)
                final_answer = self._call_llm(final_prompt)
                return final_answer
            elif "Yes" in followup_response and "Follow up:" in followup_response:
                # 提取子问题
                followup_question = followup_response.split("Follow up:")[-1].strip()
                self.history.append({"followup": followup_question, "answer": ""})
                print(f"子问题 {step}: {followup_question}")

                # 3. 搜索子问题答案（失败则用LLM回答）
                search_answer = self._search(followup_question)
                if search_answer:
                    intermediate_answer = search_answer
                else:
                    print(f"搜索无结果，使用LLM回答子问题：{followup_question}")
                    intermediate_answer = self._call_llm(followup_question)

                # 4. 记录中间答案，更新历史
                self.history[-1]["answer"] = intermediate_answer
                current_history += f"\nFollow up:{followup_question}\nIntermediate answer: {intermediate_answer}"
                print(f"中间答案 {step}: {intermediate_answer}")
            else:
                # LLM输出格式异常，直接用LLM回答原问题
                print("LLM输出格式异常，直接回答原问题")
                return self._call_llm(query)

        # 超过最大步骤，强制生成最终答案
        final_prompt = self._get_final_answer_prompt(query, current_history)
        return self._call_llm(final_prompt)


# 示例运行
if __name__ == "__main__":
    # 初始话模型调用客户端
    client = initialize_model_client()
    # 初始化代理
    agent = SelfAskAgent(client=client, max_steps=5)

    # 测试2跳问题
    question = "超导被发现时，谁是美国总统？"
    print(f"问题：{question}")
    answer = agent.run(question)
    print(f"最终答案：{answer}")

    #  测试4跳问题
    print("-" * 50)
    question = "北京奥运会那年谁赢得了大师赛？,这个赢得了大师赛的人出生的那年谁是美国总统"
    print(f"\n问题：{question}")
    answer = agent.run(question)
    print(f"最终答案：{answer}")
