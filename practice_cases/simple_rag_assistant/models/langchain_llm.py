"""
LangChain LLM 封装模块
适配兼容OpenAI接口规范的模型：千问(Qwen)、DeepSeek、OpenAI、智谱(Zhipu)等
"""

# 导入必要的Python库
import os  # 用于处理操作系统相关的功能
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 模型配置常量（集中管理，便于维护）
MODEL_CONFIG_MAP: Dict[str, Dict[str, str]] = {
    "qwen": {
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "default_model": "qwen-plus"
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_model": "deepseek-chat"
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_model": "gpt-4-turbo"
    },
    "zhipu": {
        "api_key_env": "ZHIPU_API_KEY",
        "base_url_env": "ZHIPU_BASE_URL",
        "default_model": "glm-4"
    }
}


def _get_env_var(env_name: str, model_type: str) -> str:
    """
    安全获取并校验环境变量，提供错误提示。
    Args:
        env_name: 待读取的环境变量名称（如QWEN_API_KEY）
        model_type: 模型类型标识（如qwen/deepseek），用于错误提示上下文

    Returns:
        str: 非空的环境变量值

    Raises:
        ValueError: 当环境变量未设置或值为空时抛出，包含明确的配置指引
    """
    value = os.getenv(env_name)
    if not value:
        raise ValueError(
            f"[{model_type.upper()}] 缺少必要的环境变量：{env_name}\n"
            f"请在.env文件中配置 {env_name}=你的API密钥/基础地址"
        )
    return value


def langchain_llm(
        model_type: str = "qwen",
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs
) -> BaseChatModel:
    """
    统一的LLM模型初始化入口函数，适配所有兼容OpenAI接口规范的模型。
    已支持模型：qwen（通义千问）/deepseek(深度求索)/openai(OpenAI官方)/zhipu(智谱GLM模型)，可配置MODEL_CONFIG_MAP扩展。

    Args:
        model_type: 模型类型标识，支持：qwen/deepseek/openai/zhipu
        model: 具模型名称，不传则使用默认值
        temperature: 生成温度系数，控制输出随机性（0=完全确定，1=高度随机），默认0.0
        **kwargs: 透传参数，会传递给底层的init_chat_model/ChatOpenAI初始化方法
                  支持的参数示例：max_tokens(生成最大长度)、timeout(超时时间)、top_p(采样阈值)等

    Returns:
        BaseChatModel: 初始化完成的LangChain聊天模型实例，可直接用于对话生成
    """
    # 校验模型类型是否支持
    if model_type not in MODEL_CONFIG_MAP:
        raise ValueError(
            f"不支持的模型类型：{model_type}\n"
            f"当前支持的类型：{list(MODEL_CONFIG_MAP.keys())}"
        )

    # 获取模型配置
    config = MODEL_CONFIG_MAP[model_type]
    model = model or config["default_model"]

    # 获取环境变量
    api_key = _get_env_var(config["api_key_env"], model_type)
    base_url = _get_env_var(config["base_url_env"], model_type)

    # 根据模型类型初始化
    if model_type == "deepseek":
        # DeepSeek使用init_chat_model初始化
        llm = init_chat_model(
            model=model,
            api_key=api_key,
            api_base=base_url,
            temperature=temperature,
            model_provider="deepseek",
            **kwargs
        )
    else:
        # 其他模型使用ChatOpenAI（兼容OpenAI接口）
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            **kwargs
        )

    return llm


def langchain_qwen_llm(model: str = "qwen-plus", temperature: float = 0.0) -> BaseChatModel:
    """初始化千问聊天模型"""
    return langchain_llm("qwen", model=model, temperature=temperature)


def langchain_deepseek_llm(model: str = "deepseek-chat", temperature: float = 0.0) -> BaseChatModel:
    """初始化DeepSeek聊天模型"""
    return langchain_llm("deepseek", model=model, temperature=temperature)


def langchain_openai_llm(model: str = "gpt-4-turbo", temperature: float = 0.0) -> BaseChatModel:
    """初始化OpenAI聊天模型"""
    return langchain_llm("openai", model=model, temperature=temperature)


# 新增智谱初始化函数（扩展支持）
def langchain_zhipu_llm(model: str = "glm-4", temperature: float = 0.0) -> BaseChatModel:
    """初始化智谱聊天模型"""
    return langchain_llm("zhipu", model=model, temperature=temperature)


if __name__ == "__main__":
    print("=" * 50)
    print("开始测试模型初始化与调用")
    print("=" * 50 + "\n")

    user_query = "请用3句话介绍下自己"
    print(f"用户提问：{user_query}")

    # 1. 测试千问模型
    print("【测试千问模型...】")
    qwen_llm = langchain_qwen_llm(model="qwen-plus", temperature=0)
    response = qwen_llm.invoke(user_query)
    print(f"千问响应：\n{response.content}\n")

    # 2. 测试DeepSeek模型
    print("【测试 DeepSeek 模型...】")
    deepseek_llm = langchain_deepseek_llm("deepseek-chat", temperature=0)
    response = deepseek_llm.invoke(user_query)
    # 美化输出响应
    print(f"DeepSeek响应：\n{response.content}\n")

    # 3. 测试OpenAI模型（保留注释，需启用时取消注释即可）
    # print("【测试 OpenAI 模型...】")
    # 取消下面2行注释即可启用OpenAI测试
    # openai_llm = initialize_openai_llm(model="gpt-5", temperature=0.3)
    # print("ℹ️  若需测试OpenAI，取消代码中OpenAI相关的注释即可\n")
