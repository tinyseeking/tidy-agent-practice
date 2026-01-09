from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from typing import Dict, List, Any, Type

from pydantic import BaseModel, Field

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


# 使用pydantic定义数据模型
class TravelDetails(BaseModel):
    destination: str = Field(description="目的地名称")
    duration: int = Field(description="旅行天数")


class DestinationSuggestions(BaseModel):
    destinations: List[str] = Field(description="推荐的目的地列表")


class HotelSuggestions(BaseModel):
    hotels: List[str] = Field(description="推荐的酒店列表")


def extract_structured_prompt(model_class: BaseModel) -> str:
    """提取结构化数据提示"""
    # 构建包含JSON Schema的系统提示
    # ensure_ascii=False 确保非 ASCII 字符正确输出
    # indent=2 增加 JSON 字符串的可读性
    structured_prompt = (
        "根据用户描述生成回复，并严格按照以下JSON Schema返回JSON对象：\n"
        f"{json.dumps(model_class.model_json_schema(), ensure_ascii=False, indent=2)}\n\n"
        "重要规则：\n"
        "1. 只返回符合Schema的JSON对象，不添加任何额外内容或注释\n"
        "2. 所有required字段必须包含且类型正确\n"
        "3. 数值字段必须为数字类型，而非字符串\n"
        "4. 日期应使用ISO 8601格式（YYYY-MM-DD）\n"
        "5. 布尔值必须为true/false，而非字符串\n\n"
    )
    return structured_prompt


def generate_trip_details(query: str) -> TravelDetails | dict[Any, Any]:
    """提取旅行基本信息"""
    # 构建包含TravelDetails的 JSON Schema的系统提示
    system_prompt = extract_structured_prompt(TravelDetails)
    # 构建请求消息
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    # 调用模型生成回复
    response = client.chat.completions.create(
        model="qwen-plus",  # 模型名称
        messages=input_messages,  # 对话消息列表（格式：[{"role": "user/system/assistant", "content": "内容"}]）
        temperature=0.1,   # 随机性控制参数：0~1，值越低输出越稳定/确定，值越高越具创造性（推荐0.1~0.5）
        response_format={"type": "json_object"}  # 强制模型输出标准JSON格式，便于后续解析
    )
    return json.loads(response.choices[0].message.content)


def suggest_destinations(trip_details: Dict[str, Any]) -> Dict[str, List[str]]:
    """生成推荐目的地列表"""

    # 构建包含DestinationSuggestions 的JSON Schema的系统提示
    system_prompt = extract_structured_prompt(DestinationSuggestions)

    query = f" 建议本次旅行的 3-5 个目的地。 旅行详情如下: {json.dumps(trip_details)}"
    # 构建请求消息
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    # 调用模型生成回复
    response = client.chat.completions.create(
        model="qwen-plus",  # 模型名称
        messages=input_messages,  # 对话消息列表（格式：[{"role": "user/system/assistant", "content": "内容"}]）
        temperature=0.3,  # 随机性控制参数：0~1，值越低输出越稳定/确定，值越高越具创造性（推荐0.1~0.5）
        response_format={"type": "json_object"}  # 强制模型输出标准JSON格式，便于后续解析
    )
    # 解析模型返回的JSON字符串为Python字典并返回
    return json.loads(response.choices[0].message.content)


def find_hotels(suggestions: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """查找推荐目的地的酒店"""

    destinations = suggestions["destinations"]
    # 构建包含HotelSuggestions 的JSON Schema的系统提示
    system_prompt = extract_structured_prompt(HotelSuggestions)

    query = f" 为每个目的地推荐 1-2 家热门酒店。目的地如下: {json.dumps(destinations)}"
    # 构建请求消息
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    # 调用模型生成回复
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=input_messages,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def create_itinerary(
        trip_details: Dict[str, Any],
        suggestions: Dict[str, List[str]],
        hotels: Dict[str, List[str]]
) -> str:
    """创建详细行程计划"""
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system",
             "content": "制定详细的旅行行程，包括航班、酒店、每日活动和预算估算。"},
            {"role": "user", "content": f"""为前往{trip_details['destination']}的旅行创建全面的5天行程安排.

            旅行详情：
            - 时长：{trip_details['duration']}天
            - 目的地：{', '.join(suggestions['destinations'])}
            - 酒店：{', '.join(hotels['hotels'])}
            需包含：
            1. 带活动安排的每日时间表
            2. 每夜的酒店推荐
            3. 城市间的交通选择
            4. 每日预算估算
            5. 文化小贴士和当地体验
            """}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


async def generate_trip_plan(query: str) -> str:
    """生成完整旅行计划的异步主函数"""
    # 1. 提取旅行基本信息
    trip_details = generate_trip_details(query)
    print(f"提取的旅行信息: {trip_details}")

    # 2. 生成推荐目的地
    suggestions = suggest_destinations(trip_details)
    print(f"推荐的目的地: {suggestions}")

    # 3. 查找酒店
    hotels = find_hotels(suggestions)
    print(f"推荐的酒店: {hotels}")

    # 4. 创建详细行程
    itinerary = create_itinerary(trip_details, suggestions, hotels)
    return itinerary


if __name__ == "__main__":
    import asyncio

    # 用户查询
    user_query = "日本5日旅行计划"

    trip_plan = asyncio.run(generate_trip_plan(user_query))
    print(f"\n=== {user_query} ===")
    print(trip_plan)
