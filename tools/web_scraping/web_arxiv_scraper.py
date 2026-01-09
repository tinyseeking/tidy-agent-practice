import arxiv


def search_arxiv(query: str, max_results: int = 10,
                 sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate) -> list:
    """
    在arXiv上搜索学术论文

    参数:
        query: 搜索查询关键词
        max_results: 最大返回结果数
        sort_by: 排序标准 (SubmittedDate, LastUpdatedDate, Relelvance)

    返回:
        包含论文信息的字典列表
    """
    try:
        # 执行搜索
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )

        # 处理搜索结果
        results = []
        for result in search.results():
            paper_info = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url
            }
            results.append(paper_info)

        return results

    except Exception as e:
        print(f"搜索过程中发生错误: {e}")
        return []


def print_paper_info(papers: list) -> None:
    """
    格式化打印论文信息

    参数:
        papers: 包含论文信息的字典列表
    """
    if not papers:
        print("未找到相关论文")
        return

    for i, paper in enumerate(papers, 1):
        print(f"论文 {i}:")
        print(f"  标题: {paper['title']}")
        print(f"  作者: {', '.join(paper['authors'])}")
        print(f"  发布时间: {paper['published']}")
        print(f"  PDF链接: {paper['pdf_url']}")
        print(f"  摘要: {paper['summary'][:200]}...\n")  # 显示摘要前200个字符


# 测试调用
if __name__ == "__main__":
    # 定义测试查询
    test_query = "ai agent"

    print(f"正在搜索关于 '{test_query}' 的论文...\n")
    papers = search_arxiv(test_query, max_results=5)

    # 打印搜索结果
    print_paper_info(papers)
