# 导入必要的Python库
import os  # 用于处理操作系统相关的功能
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, ZhipuAIEmbeddings

from llm.langchain.dashscope_embedding import DashScopeEmbeddings

# 加载环境变量
load_dotenv()


def langchain_embedding_model(provider: str = "qwen"):
    """
        初始化并返回指定提供商的嵌入模型

        参数:
            provider (str): 嵌入模型提供商，支持"openai"、"huggingface"、"qwen" 或 “zhipu”,默认为 qwen

        返回:
            embeddings: 初始化后的嵌入模型实例
        """
    # 加载环境变量
    if provider.lower() == "openai":
        # 使用OpenAI的嵌入模型
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("缺少OPENAI_API_KEY环境变量")

        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return OpenAIEmbeddings(
                openai_api_key=api_key,
                base_url=base_url,
                model="text-embedding-ada-002"
            )
        else:
            return OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-ada-002"
            )

    elif provider.lower() == "huggingface":
        # 使用Hugging Face的嵌入模型
        model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL",
                               "sentence-transformers/all-MiniLM-L6-v2")

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # 可根据需要改为"cuda"使用GPU
            encode_kwargs={"normalize_embeddings": True}
        )

    elif provider.lower() == "qwen":
        # 使用千问的嵌入模型
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("缺少QWEN_API_KEY环境变量")

        base_url = os.getenv("QWEN_BASE_URL")
        if not base_url:
            raise ValueError("缺少QWEN_BASE_URL环境变量")

        return DashScopeEmbeddings(
            dashscope_api_key=api_key,

            model="text-embedding-v4",  # 千问嵌入模型名称，根据实际情况调整
        )
    elif  provider.lower() == "zhipu":
        # 使用智谱清言的嵌入模型
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("缺少ZHIPU_API_KEY环境变量")
        return ZhipuAIEmbeddings(
             model="embedding-3",
             api_key=os.getenv("ZHIPU_API_KEY")
        )

    else:
        raise ValueError(f"不支持的嵌入模型提供商: {provider}。请选择'openai'、'huggingface'或'qwen'")


def test_embedding_model(provider: str = "openai"):
    """测试嵌入模型的基本功能"""
    try:
        # 初始化嵌入模型
        embeddings = langchain_embedding_model(provider)

        # 测试文本
        test_text = "这是一个测试文本，用于验证嵌入模型的功能。"

        # 生成嵌入向量
        vector = embeddings.embed_query(test_text)

        # 打印基本信息
        print(f"使用 {provider.upper()} 嵌入模型生成向量成功")
        print(f"向量维度: {len(vector)}")
        print(f"向量前10个元素: {vector[:10]}")

        return vector

    except Exception as e:
        print(f"测试失败: {str(e)}")
        return None


if __name__ == "__main__":
    # 测试千问模型
    print("\n=== 测试千问嵌入模型 ===")
    test_embedding_model("qwen")

    # 测试OpenAI模型
    # print("=== 测试OpenAI嵌入模型 ===")
    # test_embedding_model("openai")
