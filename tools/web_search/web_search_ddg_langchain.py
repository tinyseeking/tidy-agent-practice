from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()
result = search.invoke("AI 最新发展")
print(result)

