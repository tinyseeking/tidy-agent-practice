import requests
from bs4 import BeautifulSoup


def fetch_html_content(url, headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}):
    """
    抓取指定 URL 的 HTML 文本内容

    参数:
        url (str): 要抓取的网页 URL

    返回:
        dict: 包含状态码、HTML 文本和提取的纯文本
    """
    try:
        # 发送 HTTP 请求

        response = requests.get(url, headers=headers, timeout=10)

        # 检查状态码
        response.raise_for_status()

        # 获取 HTML 内容
        html_content = response.text

        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取纯文本 (去除 HTML 标签)
        text_content = soup.get_text(separator='\n', strip=True)

        return text_content

    except requests.exceptions.RequestException as e:
        return '抓取失败'


# 使用示例
if __name__ == "__main__":
    #url = "https://36kr.com/"
    url = "https://www.infoq.cn"
    result = fetch_html_content(url)
    print(result)