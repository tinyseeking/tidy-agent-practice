from typing import List, Union, Tuple, Type, TypeVar, Optional
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm.langchain.langchain_llm import langchain_qwen_llm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义泛型类型变量
T = TypeVar('T', bound=BaseModel)


def langchain_llm_structured_output(
        llm: BaseChatModel,
        messages: Union[List[Tuple[str, str]], List[BaseMessage]],
        response_model: Type[T]
) -> Optional[T]:
    """
    调用langchain LLM并获取结构化输出

    Args:
        llm: 已初始化的LangChain聊天模型
        messages: 消息列表，可为元组列表(角色, 内容)或BaseMessage列表
        response_model: Pydantic模型类，用于定义输出结构

    Returns:
        符合response_model结构的实例，失败时返回None
    """

    try:
        # 创建解析器生成格式说明
        parser = PydanticOutputParser(pydantic_object=response_model)

        # 格式化输入消息
        formatted_messages = _format_messages(messages)

        # 创建提示模板（包含千问需要的明确格式说明）
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是专业助手，必须严格按照以下格式要求返回数据，不要添加任何额外内容：\n"
             "{format_instructions}\n"
             "请确保输出完全符合上述格式，字段类型和约束要严格遵守。"),
            *formatted_messages
        ])

        # 结合格式说明构建链，使用with_structured_output解析
        chain = prompt | llm.with_structured_output(response_model)
        # 也可替换为
        # chain = prompt | llm | parser
        # 用解析器生成格式说明替换提示中的format_instructions占位变量，并调用链
        return chain.invoke({"format_instructions": parser.get_format_instructions()})

    except Exception as e:
        logger.error(f"模型结构化解析失败: {str(e)}", exc_info=True)
        return None


def langchain_llm_structured_output_from_prompt(
        llm: BaseChatModel,
        prompt: str,
        response_model: Type[T]
) -> Optional[T]:
    """
    调用LLM并获取结构化输出，支持用户提示字符串输入

    Args:
        llm: 已初始化的LangChain聊天模型
        prompt: 用户提示字符串
        response_model: Pydantic模型类，用于定义输出结构

    Returns:
        符合response_model结构的实例，失败时返回None
    """
    # 将单提示字符串转换为标准消息列表格式，再调用主函数
    return langchain_llm_structured_output(
        llm=llm,
        messages=[("user", prompt)],  # 自动封装为用户角色消息
        response_model=response_model
    )


def _format_messages(messages: Union[List[Tuple[str, str]], List[BaseMessage]]) -> List[Tuple[str, str]]:
    """统一消息格式为(角色, 内容)元组"""
    formatted = []
    for msg in messages:
        if isinstance(msg, tuple):
            role, content = msg
            if role not in ["system", "user", "assistant"]:
                raise ValueError(f"无效的角色: {role}，必须是'system', 'user'或'assistant'")
            formatted.append((role, content))
        elif isinstance(msg, BaseMessage):
            formatted.append((msg.type, msg.content))
        else:
            raise TypeError(f"不支持的消息类型: {type(msg)}")
    return formatted


def main():

    # 测试
    class Person(BaseModel):
        """个人信息模型"""
        name: str = Field(..., description="姓名")
        age: int = Field(..., description="年龄，必须是整数")
        birth_date: str = Field(..., description="出生日期，格式为YYYY-MM-DD")
        hobbies: list[str] = Field(..., description="爱好列表，用字符串数组表示")

    # 初始化千问模型
    llm = langchain_qwen_llm(
        model="qwen-plus",
        temperature=0  # 结构化输出建议使用0温度
    )

    # 测试1: 使用消息列表方式
    messages = [("user", "我叫张三，今年28岁，1995年3月15日出生。平时喜欢打篮球、听音乐和看电影。")]
    result1 = langchain_llm_structured_output(llm, messages, Person)

    if result1:
        logger.info("=== 消息列表方式测试 ===")
        logger.info(f"解析成功:\n{result1}")
        logger.info(f"姓名: {result1.name}, 年龄: {result1.age}, 生日: {result1.birth_date}")
        logger.info(f"爱好: {', '.join(result1.hobbies)}")

    # 测试2: 使用单提示字符串方式（复用实现）
    prompt = "我叫李四，今年30岁，1993年5月20日出生。平时喜欢游泳、读书和旅行。"
    result2 = langchain_llm_structured_output_from_prompt(llm, prompt, Person)

    if result2:
        logger.info("\n=== 单提示字符串方式测试 ===")
        logger.info(f"解析成功:\n{result2}")
        logger.info(f"姓名: {result2.name}, 年龄: {result2.age}, 生日: {result2.birth_date}")
        logger.info(f"爱好: {', '.join(result2.hobbies)}")


if __name__ == "__main__":
    # 测试
    main()
