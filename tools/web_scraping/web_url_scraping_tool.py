import logging
import asyncio
from typing import List, Type

from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import BaseTool
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_default_tags() -> List[str]:
    """Get default HTML tags for web scraping"""
    return ["p", "li", "div", "a", "span", "h1", "h2", "h3", "h4", "h5", "h6"]


class WebScrapingInput(BaseModel):
    url: str = Field(description="URL to scrape")
    tags_to_extract: List[str] = Field(
        default_factory=get_default_tags, description="HTML tags to extract"
    )


class WebScrapingTool(BaseTool):
    """Scrape websites when search results are insufficient"""

    name: str = "scrape_website"
    description: str = """
        使用 Chromium 浏览器抓取完整的网站内容，以实现全面的页面提取。
        
        必填参数：
        
        url（字符串）：要抓取的完整 URL（必须包含 https:// 或 http://）
        
        可选参数：
        
        tags_to_extract（列表）：要从中提取内容的 HTML 标签
        默认值：["p", "li", "div", "a", "span", "h1", "h2", "h3", "h4", "h5", "h6"]
        自定义示例：用于代码示例的 ["pre", "code"]，用于表格的 ["table", "tr", "td"]
        
        使用场景：
        
        搜索结果不完整或不充分时
        需要包括代码示例在内的完整页面内容时
        页面包含搜索遗漏的动态 JavaScript 内容时
        需要搜索无法捕捉的特定格式或结构时
        
        示例：
        
        基本抓取：url="https://docs.langchain.com/docs/modules/agents"
        代码聚焦抓取：url="https://fastapi.tiangolo.com/tutorial/", tags_to_extract=["pre", "code", "p"]
        表格提取：url="https://docs.python.org/3/library/", tags_to_extract=["table", "tr", "td", "th"]
        
        最佳实践：
        
        仅在 search_documentation 提供的信息不足时使用
        优先选择来自先前搜索结果的 URL 以确保相关性
        对目标内容使用特定标签提取（处理速度更快）
        注意：比搜索慢约 3-10 倍，为保证性能请谨慎使用
        
        局限性：
        
        内容会在配置的限制处截断，以防止过度使用令牌
        某些网站可能会阻止自动抓取
        比搜索慢 —— 仅在搜索不足以满足需求时使用
    """
    args_schema: Type[BaseModel] = WebScrapingInput

    max_content_length: int = Field(default=20000, exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, max_content_length: int = 20000):
        super().__init__(
            max_content_length=max_content_length, args_schema=WebScrapingInput
        )

    async def _process_scraping(
        self, url: str, tags_to_extract: List[str] = None, is_async: bool = True
    ) -> str:
        """Common logic for both sync and async scraping"""
        try:
            if tags_to_extract is None:
                tags_to_extract = get_default_tags()

            loader = AsyncChromiumLoader([url])

            if is_async:
                html_docs = await asyncio.to_thread(loader.load)
            else:
                html_docs = loader.load()

            if not html_docs:
                return f"Failed to load content from {url}"

            bs_transformer = BeautifulSoupTransformer()

            if is_async:
                docs_transformed = await asyncio.to_thread(
                    bs_transformer.transform_documents,
                    html_docs,
                    tags_to_extract=tags_to_extract,
                )
            else:
                docs_transformed = bs_transformer.transform_documents(
                    html_docs,
                    tags_to_extract=tags_to_extract,
                )

            if not docs_transformed:
                return f"No content extracted from {url}"

            content = docs_transformed[0].page_content

            if len(content) > self.max_content_length:
                content = (
                    content[: self.max_content_length] + "\n\n... (content truncated)"
                )

            return f"""
                    **Website Scraped:** {url}
                    **Content Extracted:**
                    
                    {content}
                    
                    **Note:** Complete website content for comprehensive analysis.
                    """

        except Exception as e:
            return f"Web scraping error for {url}: {str(e)}"

    def _run(self, url: str, tags_to_extract: List[str] = None) -> str:
        """Scrape website content"""
        return asyncio.run(
            self._process_scraping(url, tags_to_extract, is_async=False)
        )

    async def _arun(self, url: str, tags_to_extract: List[str] = None) -> str:
        """Async version of scraping"""
        return await self._process_scraping(url, tags_to_extract, is_async=True)
