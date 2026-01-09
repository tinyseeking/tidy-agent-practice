import asyncio
import re
from bs4 import BeautifulSoup, Tag, NavigableString
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

# 默认用户代理（模拟Chrome浏览器）
DEFAULT_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36")


def clean_whitespace(text: str) -> str:
    """
    清理文本中的多余空白，同时保留基本格式

    参数:
        text: 需要清理的文本

    返回:
        清理后的文本
    """
    # 替换多个换行符为两个换行符（保留段落分隔）
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # 替换行内多个空格为一个空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 去除每行首尾的空白
    lines = [line.strip() for line in text.split('\n')]
    # 过滤空行
    lines = [line for line in lines if line]
    return '\n'.join(lines)


def extract_full_content(soup: BeautifulSoup) -> str:
    """
    从BeautifulSoup对象中提取完整网页内容，尽量保持原文结构

    参数:
        soup: 解析后的BeautifulSoup对象

    返回:
        保持原始结构的网页文本内容
    """
    # 尝试找到主要内容区域（许多网站使用main标签）
    main_content = soup.find('main')

    # 如果找不到main标签，使用body标签
    if not main_content:
        main_content = soup.find('body')

    if not main_content:
        return "无法提取网页内容"

    content_parts = []

    # 遍历主要内容区域的所有子元素
    for element in main_content.descendants:
        if isinstance(element, Tag):
            # 处理标题标签
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # 标题前后添加空行，突出显示
                level = int(element.name[1])
                content_parts.append('\n' + '#' * level + ' ' + element.get_text(strip=True) + '\n')

            # 处理段落标签
            elif element.name == 'p':
                text = element.get_text()
                if text.strip():  # 只添加非空段落
                    content_parts.append(text + '\n')

            # 处理列表
            elif element.name in ['ul', 'ol']:
                # 在列表前后添加空行
                content_parts.append('\n')
                # 遍历列表项
                for li in element.find_all('li', recursive=False):
                    marker = '- ' if element.name == 'ul' else '1. '
                    content_parts.append(marker + li.get_text().strip() + '\n')
                content_parts.append('\n')

            # 处理表格
            elif element.name == 'table':
                content_parts.append('\n[表格开始]\n')
                # 提取表格内容
                for row in element.find_all('tr'):
                    cells = row.find_all(['th', 'td'])
                    if cells:
                        row_content = ' | '.join(cell.get_text(strip=True) for cell in cells)
                        content_parts.append(row_content + '\n')
                content_parts.append('[表格结束]\n\n')

            # 处理图片
            elif element.name == 'img':
                alt_text = element.get('alt', '无描述图片')
                src = element.get('src', '无图片地址')
                content_parts.append(f'\n[图片: {alt_text}] (地址: {src})\n')

            # 处理链接
            elif element.name == 'a' and 'href' in element.attrs:
                link_text = element.get_text(strip=True) or '链接'
                href = element['href']
                content_parts.append(f'[{link_text}]({href})')

        # 处理纯文本节点（非标签）
        elif isinstance(element, NavigableString):
            # 确保是有实际内容的文本
            text = str(element).strip()
            if text:
                # 避免重复添加已通过标签处理的文本
                if element.parent and isinstance(element.parent, Tag):
                    parent_tag = element.parent.name
                    if parent_tag not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'th', 'td', 'a']:
                        content_parts.append(str(element))

    # 合并所有部分并清理空白
    full_content = ''.join(content_parts)
    return clean_whitespace(full_content)


async def fetch_html_content(url: str, timeout: int = 120000) -> str:
    """
    获取网页并提取完整内容，尽量保持原文格式和结构

    参数:
        url: 目标网页URL
        timeout: 超时时间(毫秒)

    返回:
        包含网页信息和完整内容的字典，失败则返回None
    """
    # 验证URL格式
    if not (url.startswith('http://') or url.startswith('https://')):
        ValueError(f"无效的URL格式: {url}，必须以http://或https://开头")
        return ""

    try:

        async with async_playwright() as p:
            # 启动浏览器（无头模式）
            browser = await p.chromium.launch(headless=True,
                                              args=[
                                                  "--disable-gpu",
                                                  "--no-sandbox",  # 提高兼容性
                                                  "--disable-dev-shm-usage",
                                                  f"--user-agent={DEFAULT_USER_AGENT}"
                                              ]
                                              )
            page = await browser.new_page()
            # 设置页面超时时间
            page.set_default_timeout(timeout)

            # 导航到目标URL，等待网络稳定（500ms内没有网络请求）
            # 这比单纯的DOMContentLoaded更可靠，能确保动态内容加载完成
            #await page.goto(url, wait_until="networkidle")
            await page.goto(url)
            await asyncio.sleep(2)  # 额外等待JavaScript渲染
            # 获取完整页面内容
            html_content = await page.content()
        # 解析HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取保持格式的完整内容
        full_content = extract_full_content(soup)

        return full_content

    except PlaywrightTimeoutError:
        print("错误：页面加载超时，未能获取搜索结果")
        return ""
    except Exception as e:
        print(f"获取搜索结果时发生错误：{str(e)}")
        return ""
    finally:
        # 确保浏览器关闭
        await browser.close()


async def main():
    # 获取用户查询
    url = "https://www.infoq.cn"
    # url = "https://www.21jingji.com/article/20250620/herald/7f1afe1c604df468dc0cf669fe497904.html"

    print(f"正在获取URL内容：{url}...")

    # 获取搜索结果
    page_content = await fetch_html_content(url)

    print(f"搜索结果：{page_content}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已被用户中断")
    except Exception as e:
        print(f"程序运行出错：{str(e)}")
