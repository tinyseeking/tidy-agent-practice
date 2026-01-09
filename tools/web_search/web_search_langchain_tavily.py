import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
# 加载环境变量
load_dotenv()

# 确保API密钥已设置
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = ''


# 实例化工具
tool = TavilySearch(
    max_results=10,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# 进行搜索
search_query = "AI 编程发展趋势"
result = tool.invoke({"query": search_query})

print(f"结果： \n {result}")
# 打印搜索结果
for result in result['results']:
    print("*************************")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Content: {result['content']}\n")
    print(f"score: {result['score']}\n")
   # print(f"raw_content: {result['raw_content']}\n")
