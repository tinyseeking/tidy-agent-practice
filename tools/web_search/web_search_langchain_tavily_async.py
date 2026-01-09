import os

from dotenv import load_dotenv

from tools.web_search.langchain_tavily_search import TavilySearch

# 加载环境变量
load_dotenv()

# 确保API密钥已设置
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = ''


async def search():
    # 实例化工具
    search_tool = TavilySearch(
        max_results=10,
        search_depth="advanced",
        # include_answer=True,
        # include_raw_content=True,
        # include_images=True,
    )

    search_query = "2024年诺贝尔物理学奖得主"
    # 异步调用
    result= await search_tool.ainvoke(search_query)

    print(f"结果： \n ")
    # 打印搜索结果
    for result in result['results']:
        print("*************************")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content: {result['content']}\n")
        print(f"score: {result['score']}\n")
        print(f"raw_content: {result['raw_content']}\n")

if __name__ == "__main__":
    import asyncio

    asyncio.run(search())




