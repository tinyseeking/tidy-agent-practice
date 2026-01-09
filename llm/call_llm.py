import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import Client, OpenAI
from openai.types.chat import ChatCompletionMessage

# 加载环境变量
load_dotenv()

# 模型配置映射表，包含各模型所需的环境变量和默认参数
MODEL_CONFIG: Dict[str, Dict[str, str]] = {
    # 千问模型配置
    # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    "qwen": {
        "api_key_env": "QWEN_API_KEY",  # 配置获取模型API KEY 的环境变量
        "base_url_env": "QWEN_BASE_URL",  # 配置获取模型API URL 的环境变量
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 默认调用地址
        "default_model": "qwen-plus-latest"  # 默认调用模型名称
    },
    # zhipu模型配置
    "zhipu": {
        "api_key_env": "ZHIPU_API_KEY",
        "base_url_env": "ZHIPU_BASE_URL",
        "default_base_url": None,
        "default_model": "GLM-4.5-Air"
    },
    # deepseek模型配置
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": None,
        "default_model": "deepseek-chat"
    },
    # openai模型配置
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": None,
        "default_model": "gpt-4-turbo"
    },
}


def init_model_client(provider: str = "qwen") -> Client:
    """
     根据模型类型获取对应的openai客户端,默认使用千问模型

     Args:
         provider: 模型类型，当前支持 "qwen", "openai", "deepseek","zhipu"

     Returns:
         OpenAI客户端实例
     """
    if provider not in MODEL_CONFIG:
        raise ValueError(f"不支持的模型类型: {provider}，支持的类型有: {list(MODEL_CONFIG.keys())}")

    config = MODEL_CONFIG[provider]

    # 获取API密钥
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise ValueError(f"缺少{config['api_key_env']}环境变量")

    # 获取请求端口URL
    base_url = os.getenv(config["base_url_env"]) or config["default_base_url"]

    # 初始化并返回客户端
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def call_llm_chat(
        messages: List[Dict[str, str]],
        client: Client,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs  # 接受额外的生成参数
) -> ChatCompletionMessage:
    """
    通用的大模型调用函数，支持调用不同类型的模型,默认使用千问模型

    Args:
        messages: 对话消息列表，格式为[{"role": "user", "content": "消息内容"}, ...]
        client: openai客户端
        model: 模型名称，如不指定则使用对应模型类型的默认模型
        temperature: 控制模型输出随机性与多样性，默认值为 0，取值范围：0.0 ~ 1.0:
            - 0.0：生成结果最确定、一致性高（适合事实问答、逻辑推理）
            - 1.0：生成结果随机性强、更有创造性（适合文学创作、发散思维）
        **kwargs: 额外的生成参数，如top_p, max_tokens等

    Returns:
        模型返回的消息对象
    """

    # 确定使用的模型
    used_model = model or MODEL_CONFIG[provider]["default_model"]

    # 调用模型
    completion = client.chat.completions.create(
        model=used_model,
        messages=messages,
        temperature=temperature,
        **kwargs
    )

    return completion.choices[0].message


def init_qwen_client() -> Client:
    """
    实例化千问模型客户端
    """
    return init_model_client("qwen")


def call_qwen_chat(messages: list, model="qwen-plus-latest", temperature=0, **kwargs) -> ChatCompletionMessage:
    """
    调用千问模型对话
    """
    # 获取千问模型调用客户端
    client = init_qwen_client()

    # 调用千问聊天模型
    return call_llm_chat(messages=messages, client=client, model=model, temperature=temperature, **kwargs)


def init_openai_client() -> Client:
    """
    实例化openai客户端
    """

    return init_model_client("openai")


def call_openai_chat(messages: list, model="gpt-4-turbo", temperature=0, **kwargs) -> ChatCompletionMessage:
    """
    调用openai模型对话
    """
    # 获取openai客户端
    client = init_openai_client()
    # 调用openai聊天模型
    return call_llm_chat(messages=messages, client=client, model=model, temperature=temperature, **kwargs)


def init_deepseek_client() -> Client:
    """
    实例化DeepSeek模型客户端
    """
    return init_model_client("deepseek")


def call_deepseek_chat(messages: list, model="deepseek-chat", temperature=0, **kwargs) -> ChatCompletionMessage:
    """
    调用 DeepSeek模型对话
    """
    # 获取DeepSeek客户端
    client = init_deepseek_client()

    # 调用DeepSeek聊天模型
    return call_llm_chat(messages=messages, client=client, model=model, temperature=temperature, **kwargs)


def init_zhipu_client() -> Client:
    """
    实例化zhipu模型客户端
    """
    return init_model_client("zhipu")


def call_zhipu_chat(messages: list, model="GLM-4.5-Air", temperature=0, **kwargs) -> ChatCompletionMessage:
    """
    调用 zhipu模型对话
    """
    # 获取zhipu客户端
    client = init_zhipu_client()

    # 调用zhipu聊天模型
    return call_llm_chat(messages=messages, client=client, model=model, temperature=temperature, **kwargs)


if __name__ == "__main__":

    # 测试消息
    TEST_MESSAGES = [{"role": "user", "content": "请简单介绍一下你自己，用一句话回答"}]

    print("=== 测试通用调用函数 ===")
    for provider in MODEL_CONFIG.keys():
        try:
            print(f"\n测试 {provider} 模型...")
            _client = init_model_client(provider)
            response = call_llm_chat(
                messages=TEST_MESSAGES,
                client=_client,
                temperature=0,
                max_tokens=2000
            )
            print(f"返回结果: {str(response.content)}...")
        except Exception as e:
            print(f"测试失败: {str(e)}")
