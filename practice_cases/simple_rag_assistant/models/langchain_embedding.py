# 导入必要的Python库
import os  # 用于处理操作系统相关的功能

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from models.custom_dashscope_embedding import DashScopeEmbeddings

# 加载环境变量
load_dotenv()


def initialize_embedding_model(provider: str = "qwen"):
    """
        初始化并返回指定提供商的嵌入模型

        参数:
            provider (str): 嵌入模型提供商，支持"openai"、"huggingface"或"qwen",默认为 qwen

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

    elif provider.lower() == "local_bge_small":
        # BGE-small-zh-v1.5是北京智源研究院（BAAI）开发的轻量级中文文本嵌入模型，支持将文本转换为高维向量，适用于检索、分类、聚类等任务，且对资源受限场景友好。
        # 手动下载向量模型，则指定本地文件夹路径;若 SDK 自动下载，则直接用模型名。
        model_path = "../../../data/models_embedding_data/bge-small-zh-v1.5"  # 手动下载的本地路径
        # 或 model_path = "BAAI/bge-small-zh-v1.5"（直接用模型名，首次使用时，SDK 自动下载模型文件）

        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},  # 可指定 'cuda' 启用 GPU 加速
            encode_kwargs={'normalize_embeddings': True}  # 是否对输出向量归一化（推荐用于相似度计算）
        )

    else:
        raise ValueError(f"不支持的嵌入模型提供商: {provider}。请选择'openai'、'huggingface'或'qwen'")


def test_embedding_model(provider: str = "openai"):
    """测试嵌入模型的基本功能"""
    try:
        # 初始化嵌入模型
        embeddings = initialize_embedding_model(provider)

        # 测试文本
        test_text = "这是一个测试文本，用于验证嵌入模型的功能。"

        # 生成嵌入向量
        vector = embeddings.embed_query(test_text)

        # 打印基本信息
        print(f"使用 {provider.upper()} 生成向量成功")
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
    print("=== 测试bge-small-zh-v1.5嵌入模型 ===")
    test_embedding_model("local_bge_small")
