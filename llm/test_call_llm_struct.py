import os
import sys

from dotenv import load_dotenv
from openai import Client
from pydantic import BaseModel, Field


from call_llm_struct import call_llm_output_model
# 加载环境变量
load_dotenv()


# 1. 定义测试用的Pydantic模型
class Person(BaseModel):
    """个人信息模型"""
    name: str = Field(..., description="姓名")
    age: int = Field(..., description="年龄，必须是整数")
    birth_date: str = Field(..., description="出生日期，格式为YYYY-MM-DD")
    hobbies: list[str] = Field(..., description="爱好列表")


class Book(BaseModel):
    """书籍信息模型"""
    title: str = Field(..., description="书名")
    author: str = Field(..., description="作者")
    publication_year: int = Field(..., description="出版年份")
    genres: list[str] = Field(..., description="书籍类型列表")


# 2. 测试函数
def test_person_extraction():
    """测试从文本中提取个人信息"""
    try:
        # 初始化OpenAI客户端
        client = Client(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL")
        )

        # 测试文本
        query = """
        我叫张三，今年28岁，1995年3月15日出生。
        平时喜欢打篮球、听音乐和看电影。
        """

        print(f"query:{query}")
        # 提取结构化数据
        person = call_llm_output_model(
            client=client,
            query=query,
            model_class=Person,
            model="qwen-plus-latest"
        )

        print("个人信息提取结果:")
        print(f"姓名: {person.name}")
        print(f"年龄: {person.age}")
        print(f"出生日期: {person.birth_date}")
        print(f"爱好: {', '.join(person.hobbies)}")

        return person

    except Exception as e:
        print(f"测试失败: {str(e)}")


def test_book_extraction():
    """测试从文本中提取书籍信息"""
    try:
        # 初始化OpenAI客户端
        client = Client(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL")
        )

        # 测试文本
        query = """
        《三体》是刘慈欣创作的一部科幻小说，于2008年出版。
        这本书融合了硬科幻、哲学思考和社会评论等多种元素。
        """

        print(f"\n\nquery:{query}")

        # 提取结构化数据
        book = call_llm_output_model(
            client=client,
            query=query,
            model_class=Book,
            model="qwen-plus-latest"
        )

        print("\n书籍信息提取结果:")
        print(f"书名: {book.title}")
        print(f"作者: {book.author}")
        print(f"出版年份: {book.publication_year}")
        print(f"类型: {', '.join(book.genres)}")

        return book

    except Exception as e:
        print(f"测试失败: {str(e)}")


# 3. 运行测试
if __name__ == "__main__":
    test_person_extraction()
    test_book_extraction()
