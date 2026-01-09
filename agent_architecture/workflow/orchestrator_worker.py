import json
import os
import sys
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# 添加模块搜索路径，由于导入的llm模块位于当前文件的上上级目录。否则会报找不到module异常
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# 添加模块路径到sys.path中
if module_path not in sys.path:
    sys.path.append(module_path)

from llm.call_llm_struct import get_schema_json

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


# 用户查询
user_query = "写一个关于侦探的简短有趣故事"


# 定义响应模型
class OrchestratorResponse(BaseModel):
    title: str
    outline: List[str]


class WorkerResponse(BaseModel):
    paragraph: str


def process_orchestrator(user_query: str) -> OrchestratorResponse:
    """生成大纲"""
    orchestrator_prompt = (
        "你是一位专业作家。根据给定的查询生成一个故事大纲。在大纲提示中不要使用代词来指代角色。"
        "并严格按照以下JSON Schema返回JSON对象：\n"
        f"{get_schema_json(OrchestratorResponse)}\n\n"
        "注意：\n"
        "- 只返回符合Schema的JSON对象，不添加额外内容\n"
        "- 所有必需字段必须包含且类型正确\n"
        "- 日期应提取为字符串格式"
    )

    orchestrator_response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": orchestrator_prompt},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
    )

    # 解析JSON响应
    orchestrator_data = json.loads(orchestrator_response.choices[0].message.content)
    return OrchestratorResponse(**orchestrator_data)


async def process_worker_task(sub_title: str) -> str:
    """异步函数：处理单个worker任务"""
    worker_prompt = (
        "你是一位专业作家。根据大纲用自然的人类语言撰写段落。"
        "除非大纲中明确指定，否则不要假设专有名词。并严格按照以下JSON Schema返回JSON对象：\n"
        f"{get_schema_json(WorkerResponse)}\n\n"
        "注意：\n"
        "- 只返回符合Schema的JSON对象，不添加额外内容\n"
        "- 所有必需字段必须包含且类型正确\n"
        "- 日期应提取为字符串格式"
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": worker_prompt},
            {"role": "user", "content": sub_title},
        ],
        response_format={"type": "json_object"}
    )

    worker_content = response.choices[0].message.content
    worker_data = json.loads(worker_content)
    worker_model = WorkerResponse(**worker_data)
    print(f"\n 子标题： {sub_title}\n 内容：{sub_title+worker_model.paragraph}")
    return f"子标题： {sub_title}\n 内容：{sub_title+worker_model.paragraph}"


async def main():
    # 第一步：生成大纲
    orchestrator_result = process_orchestrator(user_query)

    print("Title:", orchestrator_result.title)

    # 第二步：为每个大纲项生成段落
    outline = orchestrator_result.outline

    worker_tasks = [
        process_worker_task(prompt)
        for prompt in outline
    ]
    # 收集所有段落响应
    paragraphs = await asyncio.gather(*worker_tasks)

    # 第三步：合成完整故事
    synthesizer_prompt = """
    你是一位专业作家。将段落整合成一个完整的故事。
    使用简单易懂的词汇、正确的语法和标点，保持语言流畅。
    """

    synthesizer_response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": synthesizer_prompt},
            {"role": "user", "content": "\n".join(paragraphs)},
        ],
    )

    # 解析合成响应
    story = synthesizer_response.choices[0].message.content
    print("\n完整的故事：")
    print(story)


# 运行主函数
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
