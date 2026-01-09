from duckduckgo_search import DDGS

# 执行简单的搜索
results = DDGS().text("Trends in AI Programming ", max_results=10)
print(results)