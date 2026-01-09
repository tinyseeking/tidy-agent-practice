"""Util that calls Tavily Search + Extract API.

In order to set this up, follow instructions at:
https://docs.tavily.com/docs/tavily-api/introduction
"""

import json
import os
import sys
from typing import Any, Dict, List, Literal, Optional, Union, Tuple
from urllib.parse import urlparse

import aiohttp
import requests
from aiohttp import BasicAuth, TCPConnector, ClientTimeout
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator
import logging

logger = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com"


# -----------------------------------------------------------------------
# 请求网络时，如果是同步则使用requests库发起网络请求，异步使用aiohttp发起网络请求。
# --同步调用的 requests 库会自动检测系统代理（如环境变量 HTTP_PROXY/HTTPS_PROXY、系统网络设置），
# --而异步 aiohttp 默认不会。需要手动获取系统代理并配置，实现异步网络请求。
# --所以这里对langchain默认的TavilySearchAPIWrapper做了修改添加自动获取系统代理并配置.
# ------------------------------------------------------------------------

def get_system_proxy() -> Tuple[Optional[str], Optional[BasicAuth]]:
    """
    自动检测系统代理（增强版）
    确保返回的代理URL包含协议头(http/https),
    """
    proxy_url = None
    proxy_auth = None

    # 1. 优先读取环境变量
    proxy_env_names = ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"]
    for env_name in proxy_env_names:
        if env_val := os.getenv(env_name):
            proxy_url = env_val
            logger.debug(f"从环境变量{env_name}获取代理: {proxy_url}")
            break

    # 2. 读取系统默认代理（Windows/macOS）
    if not proxy_url:
        if sys.platform == "win32":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                    r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
                    enable_proxy, _ = winreg.QueryValueEx(key, "ProxyEnable")
                    if enable_proxy:
                        proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
                        logger.debug(f"从Windows注册表获取代理服务器: {proxy_server}")

                        # 提取HTTPS代理
                        for part in proxy_server.split(";"):
                            if part.startswith("https="):
                                proxy_url = part.split("=")[1]
                                break
                            elif not part.startswith("http="):
                                proxy_url = part
            except Exception as e:
                logger.warning(f"Windows系统代理检测失败：{str(e)}")
        elif sys.platform == "darwin":
            try:
                from Foundation import CFPreferencesCopyAppValue
                if CFPreferencesCopyAppValue("HTTPSProxyEnable", "SystemConfiguration"):
                    host = CFPreferencesCopyAppValue("HTTPSProxy", "SystemConfiguration")
                    port = CFPreferencesCopyAppValue("HTTPSProxyPort", "SystemConfiguration")
                    if host and port:
                        proxy_url = f"{host}:{port}"
                        logger.debug(f"从macOS系统设置获取代理: {proxy_url}")
            except ImportError:
                logger.warning("macOS系统代理检测需安装pyobjc（执行：pip install pyobjc）")
            except Exception as e:
                logger.warning(f"macOS系统代理检测失败：{str(e)}")

    # 3. 确保代理URL包含协议头（关键修复）
    if proxy_url:
        # 检查是否已有协议头
        if not proxy_url.startswith(('http://', 'https://')):
            # 默认添加http://协议头（大多数代理使用http协议作为代理协议）
            proxy_url = f"http://{proxy_url}"
            logger.debug(f"自动为代理添加协议头: {proxy_url}")

        # 解析代理认证信息
        parsed = urlparse(proxy_url)
        if parsed.username and parsed.password:
            proxy_auth = BasicAuth(parsed.username, parsed.password)
            # 移除URL中的认证信息
            proxy_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
            logger.debug(f"提取代理认证信息，净化后的代理URL: {proxy_url}")

        logger.info(f"成功检测到代理: {proxy_url}")
    else:
        logger.info("未检测到代理，将使用直连网络")

    return proxy_url, proxy_auth


def create_proxy_session() -> aiohttp.ClientSession:
    """
    根据检测到的代理设置创建aiohttp ClientSession
    如果有代理设置，则配置对应的连接器.
    """
    # 1. 获取系统代理配置
    proxy_url, proxy_auth = get_system_proxy()
    # 2. 配置TCP连接器
    connector = TCPConnector(
        limit=10,
        ttl_dns_cache=300,
        force_close=False,
        ssl=False  # 临时禁用SSL验证（避免证书问题）
    )

    # 3. 配置会话参数
    session_kwargs: Dict[str, Any] = {
        "connector": connector,
        "connector_owner": True,
    }

    # 4. 应用代理配置（增强日志）
    if proxy_url:
        session_kwargs["proxy"] = proxy_url
        logger.debug(f"会话将使用代理: {proxy_url}")

        if proxy_auth:
            session_kwargs["proxy_auth"] = proxy_auth
            logger.debug("会话将使用代理认证")
    else:
        logger.debug("会话将不使用代理，采用直连方式")
    # 5. 创建会话
    return aiohttp.ClientSession(**session_kwargs)


class TavilySearchAPIWrapper(BaseModel):
    """Wrapper for Tavily Search API."""

    tavily_api_key: SecretStr

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
            self,
            query: str,
            max_results: Optional[int],
            search_depth: Optional[Literal["basic", "advanced"]],
            include_domains: Optional[List[str]],
            exclude_domains: Optional[List[str]],
            include_answer: Optional[Union[bool, Literal["basic", "advanced"]]],
            include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]],
            include_images: Optional[bool],
            include_image_descriptions: Optional[bool],
            topic: Optional[Literal["general", "news", "finance"]],
            time_range: Optional[Literal["day", "week", "month", "year"]],
            country: Optional[str],
    ) -> Dict:
        params = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
            "topic": topic,
            "time_range": time_range,
            "country": country,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-tavily",
        }

        response = requests.post(
            # type: ignore
            f"{TAVILY_API_URL}/search",
            json=params,
            headers=headers,
        )
        if response.status_code != 200:
            detail = response.json().get("detail", {})
            error_message = (
                detail.get("error") if isinstance(detail, dict) else "Unknown error"
            )
            raise ValueError(f"Error {response.status_code}: {error_message}")
        return response.json()

    async def raw_results_async(
            self,
            query: str,
            max_results: Optional[int],
            search_depth: Optional[Literal["basic", "advanced"]],
            include_domains: Optional[List[str]],
            exclude_domains: Optional[List[str]],
            include_answer: Optional[Union[bool, Literal["basic", "advanced"]]],
            include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]],
            include_images: Optional[bool],
            include_image_descriptions: Optional[bool],
            topic: Optional[Literal["general", "news", "finance"]],
            time_range: Optional[Literal["day", "week", "month", "year"]],
            country: Optional[str],
    ) -> Dict:
        """Get results from the Tavily Search API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
                "include_image_descriptions": include_image_descriptions,
                "topic": topic,
                "time_range": time_range,
                "country": country,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Client-Source": "langchain-tavily",
            }
            # 修改添加自动检测并配置代理
            # async with aiohttp.ClientSession() as session:
            async with create_proxy_session() as session:
                async with session.post(
                        f"{TAVILY_API_URL}/search", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)


class TavilyExtractAPIWrapper(BaseModel):
    """Wrapper for Tavily Extract API."""

    tavily_api_key: SecretStr

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
            self,
            urls: List[str],
            extract_depth: Optional[Literal["basic", "advanced"]],
            include_images: Optional[bool],
            format: Optional[str],
    ) -> Dict:
        params = {
            "urls": urls,
            "include_images": include_images,
            "extract_depth": extract_depth,
            "format": format,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-tavily",
        }

        response = requests.post(
            # type: ignore
            f"{TAVILY_API_URL}/extract",
            json=params,
            headers=headers,
        )

        if response.status_code != 200:
            detail = response.json().get("detail", {})
            error_message = (
                detail.get("error") if isinstance(detail, dict) else "Unknown error"
            )
            raise ValueError(f"Error {response.status_code}: {error_message}")
        return response.json()

    async def raw_results_async(
            self,
            urls: List[str],
            include_images: Optional[bool],
            extract_depth: Optional[Literal["basic", "advanced"]],
            format: Optional[str],
    ) -> Dict:
        """Get results from the Tavily Extract API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "urls": urls,
                "include_images": include_images,
                "extract_depth": extract_depth,
                "format": format,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Client-Source": "langchain-tavily",
            }

            # 修改添加自动检测并配置代理
            # async with aiohttp.ClientSession() as session:
            async with create_proxy_session() as session:
                async with session.post(
                        f"{TAVILY_API_URL}/extract", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)


class TavilyCrawlAPIWrapper(BaseModel):
    """Wrapper for Tavily Crawl API."""

    tavily_api_key: SecretStr

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
            self,
            url: str,
            max_depth: Optional[int],
            max_breadth: Optional[int],
            limit: Optional[int],
            instructions: Optional[str],
            select_paths: Optional[List[str]],
            select_domains: Optional[List[str]],
            exclude_paths: Optional[List[str]],
            exclude_domains: Optional[List[str]],
            allow_external: Optional[bool],
            include_images: Optional[bool],
            categories: Optional[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ],
            extract_depth: Optional[Literal["basic", "advanced"]],
            format: Optional[str],
    ) -> Dict:
        params = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "query": instructions,
            "select_paths": select_paths,
            "select_domains": select_domains,
            "exclude_paths": exclude_paths,
            "exclude_domains": exclude_domains,
            "allow_external": allow_external,
            "include_images": include_images,
            "categories": categories,
            "extract_depth": extract_depth,
            "format": format,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-tavily",
        }

        response = requests.post(
            # type: ignore
            f"{TAVILY_API_URL}/crawl",
            json=params,
            headers=headers,
        )

        if response.status_code != 200:
            detail = response.json().get("detail", {})
            error_message = (
                detail.get("error") if isinstance(detail, dict) else "Unknown error"
            )
            raise ValueError(f"Error {response.status_code}: {error_message}")
        return response.json()

    async def raw_results_async(
            self,
            url: str,
            max_depth: Optional[int],
            max_breadth: Optional[int],
            limit: Optional[int],
            instructions: Optional[str],
            select_paths: Optional[List[str]],
            select_domains: Optional[List[str]],
            exclude_paths: Optional[List[str]],
            exclude_domains: Optional[List[str]],
            allow_external: Optional[bool],
            include_images: Optional[bool],
            categories: Optional[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ],
            extract_depth: Optional[Literal["basic", "advanced"]],
            format: Optional[str],
    ) -> Dict:
        """Get results from the Tavily Crawl API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "url": url,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "instructions": instructions,
                "select_paths": select_paths,
                "select_domains": select_domains,
                "exclude_paths": exclude_paths,
                "exclude_domains": exclude_domains,
                "allow_external": allow_external,
                "include_images": include_images,
                "categories": categories,
                "extract_depth": extract_depth,
                "format": format,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Client-Source": "langchain-tavily",
            }

            # 修改添加自动检测并配置代理
            # async with aiohttp.ClientSession() as session:
            async with create_proxy_session() as session:
                async with session.post(
                        f"{TAVILY_API_URL}/crawl", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)


class TavilyMapAPIWrapper(BaseModel):
    """Wrapper for Tavily Map API."""

    tavily_api_key: SecretStr

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
            self,
            url: str,
            max_depth: Optional[int],
            max_breadth: Optional[int],
            limit: Optional[int],
            instructions: Optional[str],
            select_paths: Optional[List[str]],
            select_domains: Optional[List[str]],
            exclude_paths: Optional[List[str]],
            exclude_domains: Optional[List[str]],
            allow_external: Optional[bool],
            categories: Optional[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ],
    ) -> Dict:
        params = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "instructions": instructions,
            "select_paths": select_paths,
            "select_domains": select_domains,
            "exclude_paths": exclude_paths,
            "exclude_domains": exclude_domains,
            "allow_external": allow_external,
            "categories": categories,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-tavily",
        }

        response = requests.post(
            # type: ignore
            f"{TAVILY_API_URL}/map",
            json=params,
            headers=headers,
        )

        if response.status_code != 200:
            detail = response.json().get("detail", {})
            error_message = (
                detail.get("error") if isinstance(detail, dict) else "Unknown error"
            )
            raise ValueError(f"Error {response.status_code}: {error_message}")
        return response.json()

    async def raw_results_async(
            self,
            url: str,
            max_depth: Optional[int],
            max_breadth: Optional[int],
            limit: Optional[int],
            instructions: Optional[str],
            select_paths: Optional[List[str]],
            select_domains: Optional[List[str]],
            exclude_paths: Optional[List[str]],
            exclude_domains: Optional[List[str]],
            allow_external: Optional[bool],
            categories: Optional[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ],
    ) -> Dict:
        """Get results from the Tavily Map API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "url": url,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "instructions": instructions,
                "select_paths": select_paths,
                "select_domains": select_domains,
                "exclude_paths": exclude_paths,
                "exclude_domains": exclude_domains,
                "allow_external": allow_external,
                "categories": categories,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Client-Source": "langchain-tavily",
            }
            # 修改添加自动检测并配置代理
            # async with aiohttp.ClientSession() as session:
            async with create_proxy_session() as session:
                async with session.post(
                        f"{TAVILY_API_URL}/map", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)
