# 导入必要的Python库
import os  # 用于处理操作系统相关的功能
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv()


def langchain_llm(
        provider: str = "qwen",
        model: str = None,
        temperature=0,
        **kwargs  # 接受其他任意参数
) -> BaseChatModel:
    """
    通用LLM初始化函数
    参数:
        provider: 模型提供商，支持 "qwen", "openai", "deepseek" ,默认qwen
        model: 模型名称，若为None则使用各类型默认模型
        temperature: 生成文本的随机性/创造性参数，默认值为 0
                     取值范围：0.0 ~ 1.0
                     - 0.0：生成结果最确定、一致性高（适合事实问答、逻辑推理）
                     - 1.0：生成结果随机性强、更有创造性（适合文学创作、发散思维）
        **kwargs: 可变关键字参数，用于传递 ChatOpenAI 类的其他配置项，例如：
                  - max_tokens：单次生成的最大令牌数（控制回复长度）
    返回:
        初始化好的LLM实例
    """
    # 根据模型类型选择对应的初始化函数
    provider_llms = {
        "qwen": langchain_qwen_llm,
        "openai": langchain_openai_llm,
        "deepseek": langchain_deepseek_llm
    }

    # 检查模型类型是否支持
    if provider not in provider_llms:
        supported_types = ", ".join(provider_llms.keys())
        raise ValueError(f"不支持的模型类型: {provider}，支持的类型有: {supported_types}")

    # 获取对应的初始化函数
    initializer_llm = provider_llms[provider]

    return initializer_llm(model=model, temperature=temperature, **kwargs)


def langchain_qwen_llm(model="qwen-plus-latest", temperature=0, **kwargs) -> BaseChatModel:
    """
    初始化千问聊天模型并返回
    核心逻辑：通过 LangChain 的 ChatOpenAI 类适配千问 API（支持 OpenAI 兼容协议）

    参数:
        model (str): 模型版本名称，默认值为 "qwen-plus-latest"（千问增强版）
        temperature (float): 生成文本的随机性/创造性参数，默认值为 0
                             取值范围：0.0 ~ 1.0
                             - 0.0：生成结果最确定、一致性高（适合事实问答、逻辑推理）
                             - 1.0：生成结果随机性强、更有创造性（适合文学创作、发散思维）
        **kwargs: 可变关键字参数，用于传递 ChatOpenAI 类的其他配置项，例如：
                  - max_tokens：单次生成的最大令牌数（控制回复长度）
    """
    # 获取环境中千问 API密钥
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("缺少QWEN_API_KEY环境变量")
    base_url = os.getenv("QWEN_BASE_URL")
    if not base_url:
        raise ValueError("缺少qwen_base_url环境变量")

    # 初始化聊天模型
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
        **kwargs
    )

    return llm


def langchain_openai_llm(model="gpt-4-turbo", temperature=0, **kwargs):
    """
    初始化openai聊天模型并返回
    核心逻辑：通过 LangChain 的 ChatOpenAI 类适配（支持 OpenAI 兼容协议）

    参数:
        model (str): 模型版本名称，默认值为 "gpt-4-turbo"
        temperature (float): 生成文本的随机性/创造性参数，默认值为 0
                             取值范围：0.0 ~ 1.0
                             - 0.0：生成结果最确定、一致性高（适合事实问答、逻辑推理）
                             - 1.0：生成结果随机性强、更有创造性（适合文学创作、发散思维）
        **kwargs: 可变关键字参数，用于传递 ChatOpenAI 类的其他配置项，例如：
                  - max_tokens：单次生成的最大令牌数（控制回复长度）
    """
    # 设置openai API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("缺少OPENAI_API_KEY环境变量")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise ValueError("缺少OPENAI_BASE_URL环境变量")

    # 初始化聊天模型
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
        **kwargs
    )

    return llm


def langchain_deepseek_llm(model="deepseek-chat", temperature=0, **kwargs):
    """
    初始化DeepSeek聊天模型并返回
    核心逻辑：通过 LangChain 的 ChatOpenAI 类适配（支持 OpenAI 兼容协议）

    参数:
        model (str): 模型版本名称，默认值为 "deepseek-chat"
        temperature (float): 生成文本的随机性/创造性参数，默认值为 0
                             取值范围：0.0 ~ 1.0
                             - 0.0：生成结果最确定、一致性高（适合事实问答、逻辑推理）
                             - 1.0：生成结果随机性强、更有创造性（适合文学创作、发散思维）
        **kwargs: 可变关键字参数，用于传递 ChatOpenAI 类的其他配置项，例如：
                  - max_tokens：单次生成的最大令牌数（控制回复长度）

    """
    # 设置DeepSeek API密钥
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("缺少deepseek_api_key环境变量")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")
    if not deepseek_base_url:
        raise ValueError("缺少deepseek_base_url环境变量")

    # 初始化聊天模型
    llm = ChatDeepSeek(
        model=model,
        api_key=deepseek_api_key,
        api_base=deepseek_base_url,
        temperature=temperature,
        **kwargs
    )

    return llm
