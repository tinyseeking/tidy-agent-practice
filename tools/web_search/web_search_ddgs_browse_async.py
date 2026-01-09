import asyncio
import logging
import math

from playwright.async_api import async_playwright
from typing import List, Dict
import time


class DuckDuckGoSearch:
    """使用Playwright从DuckDuckGo获取搜索结果的爬虫"""

    def __init__(self):
        """初始化爬虫配置"""
        self.search_url = "https://duckduckgo.com/"

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        执行搜索并返回结果列表

        Args:
            query: 搜索查询字符串
            max_results: 需要获取的最大结果数

        Returns:
            包含搜索结果的列表，每个结果是一个字典，包含标题、链接、简介和发布时间
        """
        results = []

        async with async_playwright() as p:
            # 使用Chromium浏览器，设置为无头模式
            browser = await p.chromium.launch(
                headless=False,  # 不显示浏览器窗口调试
                executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"  # 替换为实际Chrome路径
            )
            page = await browser.new_page()

            try:
                # 访问DuckDuckGo主页
                await page.goto(self.search_url)

                # 等待搜索框加载完成并输入搜索词
                await page.wait_for_selector('input[name="q"]')
                await page.fill('input[name="q"]', query)
                time.sleep(3)
                await page.press('input[name="q"]', 'Enter')

                # 等待搜索结果加载
                await page.wait_for_selector('.react-results--main',timeout=120000)

                # 滚动页面以加载更多结果
                # 判断如果除数为0则向下取整，否则向上取整
                scroll_attempts = max_results // 10 if (max_results % 10) == 0 else math.ceil(max_results / 10)
                await self._scroll_page(page, scroll_attempts - 1)

                # 提取有机结果（非广告）
                result_elements = await page.query_selector_all('li[data-layout="organic"]')

                for item in result_elements[:max_results]:
                    try:
                        # 提取标题
                        title_element = await item.query_selector('h2 a')
                        title = await title_element.text_content() if title_element else None
                        link = await title_element.get_attribute('href') if title_element else None

                        # 提取摘要
                        snippet_element = await item.query_selector('[data-result="snippet"]')
                        snippet = None
                        if snippet_element:
                            snippet = await snippet_element.text_content()

                        if title and link and snippet:
                            results.append({
                                'title': title,
                                'url': link,
                                'content': snippet
                            })
                    except Exception as e:
                        print(f"处理单个结果时出错: {e}")

                    # 检查是否已达到所需结果数量
                    if len(results) >= max_results:
                        break

            except Exception as e:
                print(f"搜索过程中出错: {e}")
            finally:
                # 关闭浏览器
                await browser.close()

        return results

    async def _scroll_page(self, page, scroll_attempts=3) -> None:
        """滚动页面以加载更多内容"""
        previous_height = 0
        print(f"共翻页：{scroll_attempts}")
        for i in range(scroll_attempts):
            try:
                print(f"翻页第{i + 1}次")
                # 滚动到页面底部
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')

                # 等待页面加载更多内容
                await page.wait_for_timeout(2000)

                await page.press('button[id="more-results"]', 'Enter')

                # 检查页面是否还在加载新内容
                current_height = await page.evaluate('document.body.scrollHeight')
                if current_height == previous_height:
                    break
                previous_height = current_height

            except Exception as e:
                print(f"翻页时发生异常: {e}")
                # 发生异常时继续执行，不中断整个滚动过程
                continue


async def main():
    """主函数，演示如何使用爬虫"""
    searcher = DuckDuckGoSearch()
    query = "AI 最新进展"
    results = await searcher.search(query, max_results=15)

    # 打印搜索结果
    print(f"搜索 '{query}' 的结果：")
    print(f"共找到 {len(results)} 个结果\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   链接: {result['url']}")
        print(f"   简介: {result['content']}\n")


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
