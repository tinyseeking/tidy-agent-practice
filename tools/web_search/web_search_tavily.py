import json
import os
from typing import List, Dict

from dotenv import load_dotenv
from tavily import TavilyClient

# 加载.env文件中的环境变量
load_dotenv()


class TavilySearchClient:
    """使用Tavily API执行搜索并解析结果的工具类"""

    def __init__(self, api_key: str = None):
        """
        初始化Tavily搜索客户端

        Args:
            api_key: Tavily API密钥，如未提供则尝试从环境变量获取
        """
        # 优先使用传入的API密钥，否则尝试从环境变量获取
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

        # 验证API密钥是否存在
        if not self.api_key:
            raise ValueError("Tavily API密钥未设置，请通过参数提供或设置TAVILY_API_KEY环境变量")

        # 初始化客户端
        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, max_results: int = 10, **kwargs) -> List[Dict[str, str]]:
        """
        执行搜索并返回解析后的结果列表

        Args:
            query: 搜索查询字符串
            max_results: 需要获取的最大结果数
            **kwargs: 传递给Tavily API的其他参数

        Returns:
            包含搜索结果的列表，每个结果是一个字典，包含标题、链接、内容等信息
        """
        try:
            # 调用Tavily API执行搜索
            response = self.client.search(query, max_results=max_results, **kwargs)

            # 提取搜索结果项
            items = response.get("results", [])

            return items

        except Exception as e:
            print(f"搜索过程中发生错误: {e}")
            return []


# 使用示例
if __name__ == "__main__":

    # 初始化搜索器（从环境变量获取API密钥）
    searcher = TavilySearchClient()

    # 执行搜索
    results = searcher.search(
        query="最新AI新闻",
        max_results=10,
        search_depth="advanced",  # 可选参数
       # include_raw_content= True,# 是否包含完整内容
    )

    # 打印搜索结果
    print(f"共找到 {len(results)} 个结果\n")

    print(json.dumps(
        results,
        ensure_ascii=False,
        indent=2
    ))
    # for i, result in enumerate(results, 1):
    #     print(f"{i}. {result['title']}")
    #     print(f"   链接: {result['url']}")
    #     print(f"   摘要: {result['content']}")
    #     print(f"   score: {result['score']}")
    #     print(f"   raw_content: {result['raw_content']}")