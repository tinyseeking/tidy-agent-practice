import os
import json
import asyncio
import sys
from typing import List, Optional
from pydantic import BaseModel, Field

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


# 定义响应模型
class EvaluatorResponse(BaseModel):
    feedback: str
    possibleImprovements: List[str] = Field(default_factory=list)


# 生成故事
async def generate_story(query: str) -> str:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一位专业写作家。根据用户的输入编写故事。"},
            {"role": "user", "content": query},
        ],
        temperature=0.1
    )

    # 解析JSON响应
    content = response.choices[0].message.content
    return content


# 评估故事
async def evaluate_story(story: str) -> EvaluatorResponse:
    system_prompt = (
        "你是专注于5-10岁儿童文学的资深故事评估专家。"
        "请从故事吸引力、表述清晰度及适龄性（使用简单词汇）三个维度进行评估。  "
        "若有需要，请提供建设性反馈并给出1-2个具体改进建议；"
        "若故事已足够优秀，请回复“无需改进”。 "
        "严格按照以下JSON Schema返回JSON对象：\n"
        f"{get_schema_json(EvaluatorResponse)}\n\n"
        "注意：\n"
        "- 只返回符合Schema的JSON对象，不添加额外内容\n"
        "- 所有必需字段必须包含且类型正确\n"
        "- 日期应提取为字符串格式"
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": story},
        ],
        response_format={"type": "json_object"}
    )

    # 解析JSON响应
    content = response.choices[0].message.content
    eval_data = json.loads(content)
    eval_model = EvaluatorResponse(**eval_data)

    return eval_model


# 优化故事
async def optimize_story(
        story: str,
        feedback: str,
        possible_improvements: List[str]
) -> str:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "Optimize the story"},
            {"role": "user", "content": story},
            {"role": "user", "content": "Feedback: " + feedback},
            {"role": "user", "content": "Possible improvements: " + ", ".join(possible_improvements)},
        ],
        temperature=0.6
    )

    # 解析JSON响应
    content = response.choices[0].message.content
    return content


# 主函数
async def main():
    # 用户查询
    user_query = "写一个关于猫的趣味故事"
    story = await generate_story(user_query)
    print(f"生成故事：\n{story}")
    max_iterations = 5
    optimized_story = story

    for iteration in range(max_iterations):
        evaluation = await evaluate_story(optimized_story)
        print(f"\nIteration {iteration + 1}:")
        print("Feedback:", evaluation.feedback)
        print("Possible improvements:", evaluation.possibleImprovements)

        if not evaluation.possibleImprovements:
            print("故事已十分优秀！无需进一步优化。")
            break

        # 优化故事
        optimized_story = await optimize_story(
            optimized_story,
            evaluation.feedback,
            evaluation.possibleImprovements
        )

    print("\nFinal Optimized Story:")
    print(optimized_story)


# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())
