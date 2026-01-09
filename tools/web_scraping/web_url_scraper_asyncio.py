import logging
import asyncio
import os
from typing import List, Optional
from pydantic import BaseModel, Field

# 确保只创建一个事件循环，避免重复创建
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认用户代理（模拟Chrome浏览器）
DEFAULT_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36")


def get_default_tags() -> List[str]:
    """Get default HTML tags for web scraping"""
    return ["p", "li", "div", "a", "span", "h1", "h2", "h3", "h4", "h5", "h6"]


class WebScrapingParams(BaseModel):
    """参数模型，用于验证网页爬取的输入参数"""
    url: str = Field(description="要爬取的URL，必须包含http://或https://")
    tags_to_extract: List[str] = Field(
        default_factory=get_default_tags,
        description="要提取内容的HTML标签列表"
    )


class WebUrlScraper:
    """
    使用Chromium浏览器爬取网页内容的工具，支持动态内容提取

    主要功能:
    - 爬取包含动态JavaScript内容的网页
    - 支持自定义提取特定HTML标签的内容
    - 提供同步和异步两种调用方式
    - 内容长度限制，防止过度提取
    """

    def __init__(self, max_content_length: int = 20000):
        """
        初始化网页爬取工具

        参数:
            max_content_length: 内容的最大长度，超过将被截断
        """
        self.max_content_length = max_content_length
        self._init_playwright()  # 初始化playwright

    def _init_playwright(self):
        """初始化playwright，确保正确安装浏览器"""
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                # 确保安装了Chromium浏览器
                for browser_type in [p.chromium, p.firefox, p.webkit]:
                    try:
                        browser = browser_type.launch(headless=True)
                        browser.close()
                        break
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Playwright初始化警告: {str(e)}. 可能需要运行 'playwright install' 安装浏览器")

    async def _process_scraping(
            self,
            url: str,
            tags_to_extract: Optional[List[str]] = None
    ) -> str:
        """爬取处理的核心逻辑，仅异步实现，避免嵌套事件循环"""
        try:
            # 导入需要的库（延迟导入，仅在实际使用时加载）
            from langchain_community.document_loaders import AsyncChromiumLoader
            from langchain_community.document_transformers import BeautifulSoupTransformer

            if tags_to_extract is None:
                tags_to_extract = get_default_tags()

            # 初始化加载器
            loader = AsyncChromiumLoader([url], user_agent=DEFAULT_USER_AGENT)

            # 异步加载HTML内容 - 直接使用异步方法而非线程
            html_docs = await loader.aload()

            if not html_docs:
                return f"无法从 {url} 加载内容"

            # 转换提取内容
            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = await asyncio.to_thread(
                bs_transformer.transform_documents,
                html_docs,
                tags_to_extract=tags_to_extract,
            )

            if not docs_transformed:
                return f"无法从 {url} 提取内容"

            # 处理内容长度
            content = docs_transformed[0].page_content

            if len(content) > self.max_content_length:
                content = (
                        content[:self.max_content_length] + "\n\n... (内容已截断)"
                )

            return f"""
                    **已爬取网站:** {url}
                    **提取的内容:**
                    
                    {content}
                    
                    **注意:** 完整的网站内容用于综合分析。
                    """

        except Exception as e:
            logger.error(f"爬取 {url} 时出错: {str(e)}")
            return f"网页爬取错误 ({url}): {str(e)}"
        finally:
            # 确保所有异步资源都被正确清理
            try:
                from langchain_community.document_loaders import AsyncChromiumLoader
                await AsyncChromiumLoader.close()
            except Exception:
                pass

    def scrape(self, url: str, tags_to_extract: Optional[List[str]] = None) -> str:
        """
        同步爬取网页内容 - 使用现有事件循环而非创建新的

        参数:
            url: 要爬取的URL
            tags_to_extract: 要提取的HTML标签列表，默认为get_default_tags()返回的列表

        返回:
            提取的网页内容字符串
        """
        # 验证参数
        params = WebScrapingParams(url=url, tags_to_extract=tags_to_extract or get_default_tags())

        # 检查当前是否已有运行中的事件循环
        try:
            current_loop = asyncio.get_event_loop()
            if current_loop.is_running():
                # 如果事件循环正在运行，使用create_task
                future = asyncio.create_task(
                    self._process_scraping(params.url, params.tags_to_extract)
                )
                return current_loop.run_until_complete(future)
            else:
                # 如果事件循环未运行，直接运行
                return current_loop.run_until_complete(
                    self._process_scraping(params.url, params.tags_to_extract)
                )
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            return asyncio.run(
                self._process_scraping(params.url, params.tags_to_extract)
            )

    async def ascrape(self, url: str, tags_to_extract: Optional[List[str]] = None) -> str:
        """
        异步爬取网页内容

        参数:
            url: 要爬取的URL
            tags_to_extract: 要提取的HTML标签列表，默认为get_default_tags()返回的列表

        返回:
            提取的网页内容字符串
        """
        # 验证参数
        params = WebScrapingParams(url=url, tags_to_extract=tags_to_extract or get_default_tags())
        return await self._process_scraping(params.url, params.tags_to_extract)


# 使用示例
if __name__ == "__main__":
    # 创建爬取工具实例
    scraper = WebUrlScraper(max_content_length=10000)

    # 同步爬取示例
    try:
        result = scraper.scrape(
            url="https://www.infoq.cn",
            # url="https://www.21jingji.com/article/20250620/herald/7f1afe1c604df468dc0cf669fe497904.html"
        )
        print(result)
    except Exception as e:
        print(f"同步爬取失败: {e}")

    # 异步爬取示例
    # async def async_example():
    #     try:
    #         result = await scraper.ascrape(
    #             url="https://www.infoq.cn",
    #             tags_to_extract=["p", "a"]
    #         )
    #         print(result)
    #     except Exception as e:
    #         print(f"异步爬取失败: {e}")
    #
    #
    # asyncio.run(async_example())
