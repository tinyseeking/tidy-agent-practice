import json
from typing import Type, TypeVar
from openai import Client, OpenAIError
from pydantic import BaseModel, ValidationError

# 定义泛型类型变量，用于指定Pydantic模型类
T = TypeVar('T', bound=BaseModel)


def get_schema_json(model_class: Type[BaseModel]) -> str:
    """
    生成Pydantic模型的JSON Schema字符串

    参数:
        model_class: Pydantic模型类，用于生成对应的JSON Schema

    返回:
        str: 格式化的JSON Schema字符串
    """
    # ensure_ascii=False 确保非ASCII字符正确输出
    # indent=2 增加JSON字符串的可读性
    return json.dumps(
        model_class.model_json_schema(),
        ensure_ascii=False,
        #indent=2
    )


def call_llm_with_struction_output(
        client: Client,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0
) -> str:
    """
    调用大模型（通过OpenAI SDK）生成结构化数据响应（JSON格式）

    参数:
        client: OpenAI sdk 客户端实例
        messages: 聊天消息列表，包含系统提示和用户查询
        model: 要使用的模型名称（如"gpt-4-turbo"）
        temperature: 控制模型输出随机性与多样性，默认值为 0，取值范围：0.0 ~ 1.0:
                    - 0.0：生成结果最确定、一致性高（适合事实问答、逻辑推理）
                    - 1.0：生成结果随机性强、更有创造性（适合文学创作、发散思维）

    返回:
        str: 模型返回的JSON字符串

    异常:
        OpenAIError: 当API调用失败时抛出
        ValueError: 当API返回非预期格式结果时抛出
    """
    try:
        # 调用模型生成回复，指定JSON格式输出
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        # 验证响应内容
        if not response.choices:
            raise ValueError("API返回为空，没有生成任何内容")

        content = response.choices[0].message.content
        if not content:
            raise ValueError("API返回的消息内容为空")

        return content

    except OpenAIError as e:
        raise OpenAIError(f"OpenAI API调用失败: {str(e)}") from e


def call_llm_output_model(
        client: Client,
        query: str,
        model_class: Type[T],
        model: str
) -> T:
    """
    从用户查询中提取结构化数据并返回指定的Pydantic模型实例

    参数:
        client: OpenAI客户端实例
        query: 用户的自然语言查询
        model_class: 目标Pydantic模型类
        model: 要使用的模型名称

    返回:
        T: 提取并验证后的结构化数据
    """
    # 构建包含JSON Schema的系统提示
    system_prompt = (
        "根据用户描述生成回复，并严格按照以下JSON Schema返回JSON对象：\n"
        f"{get_schema_json(model_class)}\n\n"
        "注意：\n"
        "- 只返回符合Schema的JSON对象，不添加任何额外内容\n"
        "- 确保所有必需字段都被正确包含且类型符合要求\n"
        "- 日期应提取为字符串格式（如YYYY-MM-DD）"
    )

    # 构建请求消息列表
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    # 调用LLM获取JSON响应
    response_content = call_llm_with_struction_output(
        client=client,
        messages=input_messages,
        model=model
    )

    # 解析并验证响应
    return parse_llm_output_to_clz(
        content=response_content,
        model_class=model_class
    )


def parse_llm_output_to_clz(content: str, model_class: Type[T]) -> T:
    """
    将LLM返回的JSON字符串解析为指定的Pydantic模型实例

    参数:
        content: LLM返回的JSON字符串
        model_class: 目标Pydantic模型类

    返回:
        T: 解析后的Pydantic模型实例

    异常:
        json.JSONDecodeError: 当JSON解析失败时抛出
        ValidationError: 当解析结果不符合模型定义时抛出
    """
    try:

        # 解析JSON字符串
        data = json.loads(content)
        # 验证并创建模型实例
        return model_class(**data)

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"无法解析LLM返回的JSON: {str(e)}",
            e.doc,
            e.pos
        ) from e
    except ValidationError as e:
        raise ValidationError(
            f"LLM返回的数据不符合模型定义: {str(e)}",
            e.model,
            e.errors()
        ) from e
