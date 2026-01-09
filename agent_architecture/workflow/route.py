import os
from typing import Dict, Any, List, Optional, Callable

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


class TaskRouter:
    """AI Agent工作流中的任务路由机制实现"""

    def __init__(self, model: str = "qwen-plus"):
        self.model = model
        self.task_types = {
            "qa": "问答",
            "search": "搜索",
            "translation": "翻译",
            "writing": "写作",
            "other": "其他"
        }

        # 注册不同类型任务的处理函数
        self.task_handlers = {
            "qa": self._handle_qa,  # 处理问答请求
            "search": self._handle_search,  # 处理搜索请求
            "translation": self._handle_translation,  # 处理翻译请求
            "writing": self._handle_writing,  # 处理写作请求
            "other": self._handle_other  # 处理其它请求
        }

    def determine_task_type(self, user_input: str) -> str:
        """分析用户输入，确定任务类型"""
        prompt = f"""
        分析以下用户输入，确定最适合的任务类型。只返回以下类型之一：
        {', '.join(self.task_types.keys())}
        用户输入：{user_input}
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个任务分类助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        task_type = response.choices[0].message.content.strip().lower()

        # 确保返回的任务类型是有效的
        if task_type not in self.task_types:
            return "other"

        return task_type

    def _handle_qa(self, user_input: str) -> str:
        """处理问答类任务"""
        prompt = f"""
        作为一个知识问答助手，请回答以下问题：
        {user_input}
        请提供准确、简洁的回答。
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个知识问答助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def _handle_search(self, user_input: str) -> str:
        """处理搜索类任务"""
        prompt = f"""
        作为一个搜索助手，请为以下查询提供搜索建议：
        {user_input}
        请列出最相关的搜索关键词和可能的搜索来源。
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个搜索建议助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def _handle_translation(self, user_input: str) -> str:
        """处理翻译类任务"""
        prompt = f"""
        作为一个翻译助手，请翻译以下内容：
        {user_input}
        请提供准确的翻译。
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业翻译助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def _handle_writing(self, user_input: str) -> str:
        """处理写作类任务"""
        prompt = f"""
        作为一个写作助手，请根据以下要求创作内容：

        {user_input}

        请提供符合要求的创作内容。
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业写作助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def _handle_other(self, user_input: str) -> str:
        """处理其他类型任务"""
        prompt = f"""
        作为一个通用助手，请处理以下请求：

        {user_input}

        请提供最合适的回应。
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个通用AI助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def route_task(self, user_input: str) -> Dict[str, Any]:
        """路由任务并获取结果"""
        result = {
            "success": False,
            "user_input": user_input,
            "task_type": None,
            "task_type_name": None,
            "result": None,
            "error": None
        }

        try:
            # 确定任务类型
            task_type = self.determine_task_type(user_input)
            result["task_type"] = task_type
            result["task_type_name"] = self.task_types.get(task_type, "未知")

            # 获取对应的任务处理函数
            handler = self.task_handlers.get(task_type)

            if handler:
                # 执行任务处理
                result["result"] = handler(user_input)
                result["success"] = True
            else:
                result["error"] = f"没有找到处理 {task_type} 类型任务的处理器"

        except Exception as e:
            result["error"] = str(e)

        return result


# 使用示例
if __name__ == "__main__":
    # 创建任务路由器
    router = TaskRouter()

    # 测试用例
    test_cases = [
        "什么是机器学习？",
        "帮我搜索一下最近的科技新闻",
        "将这句话翻译成英文：人工智能正在改变世界",
        "写一篇关于气候变化的短文",
        "我今天感觉不太好"
    ]

    # 运行测试用例
    for i, test_input in enumerate(test_cases):
        print(f"\n--- 测试用例 {i + 1} ---")
        print(f"输入: {test_input}")

        result = router.route_task(test_input)

        print(f"任务类型: {result['task_type']} ({result['task_type_name']})")
        print(f"处理结果: {result['success']}")

        if result["success"]:
            print(f"输出:\n{result['result']}")
        else:
            print(f"错误: {result['error']}")
