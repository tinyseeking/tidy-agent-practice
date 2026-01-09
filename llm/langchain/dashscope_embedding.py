from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# 不同模型的批量处理大小
BATCH_SIZE = {"text-embedding-v1": 25, "text-embedding-v2": 25, "text-embedding-v3": 6, "text-embedding-v4": 6}


def _create_retry_decorator(embeddings: DashScopeEmbeddings) -> Callable[[Any], Any]:
    """创建重试装饰器，处理API调用失败的情况"""
    multiplier = 1
    min_seconds = 1  # 初始重试间隔1秒
    max_seconds = 4  # 最大重试间隔4秒
    # Wait 2^x * 1 second between each retry starting with
    # 1 seconds, then up to 4 seconds, then 4 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def embed_with_retry(embeddings: DashScopeEmbeddings, **kwargs: Any) -> Any:
    """带重试机制的嵌入生成函数，支持批量处理"""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        result = []
        i = 0
        input_data = kwargs["input"]
        input_len = len(input_data) if isinstance(input_data, list) else 1
        batch_size = BATCH_SIZE.get(kwargs["model"], 6) # 按模型获取批量大小

        # 批量处理输入，避免单次请求超出API限制
        while i < input_len:
            kwargs["input"] = (
                input_data[i: i + batch_size]
                if isinstance(input_data, list)
                else input_data
            )
            resp = embeddings.client.call(**kwargs)   # 调用嵌入API
            if resp.status_code == 200:
                result += resp.output["embeddings"]  # 提取嵌入结果
            elif resp.status_code in [400, 401]:
                raise ValueError(
                    f"status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}"
                )
            else:
                raise HTTPError(
                    f"HTTP error occurred: status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}",
                    response=resp,
                )
            i += batch_size
        return result

    return _embed_with_retry(**kwargs)


class DashScopeEmbeddings(BaseModel, Embeddings):
    """
    DashScope嵌入模型封装类，适配LangChain接口。使用该模型前，您需要先安装 dashscope Python 包，并且：
    1.将您的 API 密钥配置到环境变量 DASHSCOPE_API_KEY 中；
    2.或者，在构造函数中以命名参数的形式传入 API 密钥。
    """

    client: Any = None   # 千问嵌入API客户端
    """The DashScope client."""
    model: str = "text-embedding-v1" # 默认使用的千问嵌入模型
    dashscope_api_key: Optional[str] = None
    max_retries: int = 5   # API调用最大重试次数
    """Maximum number of retries to make when generating."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """验证环境配置，初始化千问客户端"""

        import dashscope

        # 从环境变量或参数中获取API密钥
        values["dashscope_api_key"] = get_from_dict_or_env(
            values, "dashscope_api_key", "DASHSCOPE_API_KEY"
        )
        dashscope.api_key = values["dashscope_api_key"]
        try:
            import dashscope
            # 初始化千问文本嵌入客户端
            values["client"] = dashscope.TextEmbedding
        except ImportError:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档文本生成嵌入向量（批量处理）

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = embed_with_retry(
            self, input=texts, text_type="document", model=self.model
        )
        embedding_list = [item["embedding"] for item in embeddings]
        return embedding_list

    def embed_query(self, text: str) -> List[float]:
        """为查询文本生成嵌入向量（单个文本）.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = embed_with_retry(
            self, input=text, text_type="query", model=self.model
        )[0]["embedding"]
        return embedding
