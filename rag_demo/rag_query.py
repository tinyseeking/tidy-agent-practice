import os
import sys
from typing import List

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

# 添加模块搜索路径，由于导入的llm模块位于当前文件的上级目录。否则会报找不到module异常
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# 添加模块路径到sys.path中
if module_path not in sys.path:
    sys.path.append(module_path)

# 基于langchain_community的 DashScopeEmbeddings封装，用于生成文本的向量表示.
# 使用 langchain_community 的 DashScopeEmbeddings 类向量化时，因默认批量大小超出 DashScope API 限制，导致
# 报错 batch size is invalid,it should not be larger than 10.: input.contents。故重写该类，调整默认 BATCH_SIZE 以适配限制。
from llm.langchain.dashscope_embedding import DashScopeEmbeddings
from llm.langchain.langchain_llm import langchain_qwen_llm

# 加载环境变量
load_dotenv()


# 加载文档
def load_documents(directory: str, file_types: list = None) -> list:
    """
    加载指定目录下的所有指定类型文档

    Args:
        directory: 文档所在目录路径（相对路径或绝对路径）
        file_types: 要加载的文件类型列表，如 ["pdf", "docx", "txt"]，默认加载所有支持类型

    Returns:
        加载后的文档对象列表（Document 类型，包含 page_content 和 metadata）
    """
    # 验证目录是否存在
    if not os.path.exists(directory):
        raise FileNotFoundError(f"目录不存在：{directory}")

    # 定义支持的文件类型和对应的加载器
    supported_file_types = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader
    }

    # 处理默认文件类型（加载所有支持类型）
    if file_types is None:
        file_types = list(supported_file_types.keys())

    # 存储加载的文档
    all_documents = []

    # 遍历每种文件类型，加载对应文档
    for file_type in file_types:
        loader = DirectoryLoader(
            path=directory,
            glob=f"**/*.{file_type}",  # 递归匹配所有子目录下的对应类型文件
            loader_cls=supported_file_types[file_type],
            show_progress=True,  # 显示加载进度（可选）
            use_multithreading=True  # 多线程加载（提高效率，可选）
        )

        try:
            docs = loader.load()
            print(f"成功加载 {len(docs)} 个 {file_type} 文档")
            all_documents.extend(docs)
        except Exception as e:
            print(f"加载 {file_type} 文档时出错：{str(e)}")
            continue

    return all_documents


# 文本分块，使用递归字符分割器，平衡语义完整性和检索效率
def split_documents(documents, chunk_size=512, chunk_overlap=200):
    """将文档分割成小块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 每个片段的字符数（根据模型上下文调整）
        chunk_overlap=chunk_overlap,  # 片段间重叠字符数（保持上下文连贯性）
        separators=["\n\n", "\n", "。", " ", ""]  # 优先按中文标点分割，提升分块合理性
    )
    return text_splitter.split_documents(documents)


def init_dashscope_embedding_model(model="text-embedding-v4") -> Embeddings:
    """
    初始化嵌入模型,用于生成文本的向量表示。这里使用千问向量模型，需配置 QWEN_API_KEY 和 QWEN_BASE_URL 环境变量

    参数:
        model (str): 千问嵌入模型版本名称，默认使用最新版模型 text-embedding-v4，可指定其它千问嵌入模型（如 text-embedding-v3）
                     支持的模型版本需参考阿里云千问官方文档：https://help.aliyun.com/document_detail/2793376.html
    返回:
        DashScopeEmbeddings: 千问嵌入模型实例,可直接用于 LangChain 生态中的向量生成（如调用 embed_query/embed_documents 方法）
    """
    # 从环境变量读取千问 API 密钥
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("缺少 QWEN_API_KEY 环境变量（从阿里云控制台获取）")

    # 从环境变量读取千问 API 服务地址
    # base_url = os.getenv("QWEN_BASE_URL")
    # if not base_url: # "缺少 QWEN_BASE_URL 环境变量，默认地址：https://dashscope.aliyuncs.com/api/v1")
    #     base_url="https://dashscope.aliyuncs.com/api/v1"

    # 初始化并返回千问嵌入模型实例
    return DashScopeEmbeddings(
        dashscope_api_key=api_key,  # 传入API密钥用于身份验证
        model=model,  # 指定使用的嵌入模型版本
        # base_url=base_url  # 传入服务地址
    )


# 创建向量数据库并持久化存储
def create_and_persist_vectorstore(docs, embedding_model="text-embedding-v4", persist_dir="./vectorstore"):
    """创建向量数据库并持久化"""
    embeddings = init_dashscope_embedding_model(model=embedding_model)

    # 初始化Chroma向量库（若目录存在则复用，重置后添加新文档）
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.reset_collection()  # 清空现有数据
    vectorstore.add_documents(docs)  # 添加新文档

    return vectorstore


# 定义自定义提示模板：约束 LLM 基于检索结果生成答案
def get_prompt_template():
    """定义提示模板，指导LLM如何使用检索结果"""
    prompt_template = """
    你是一个专业的领域问答助手,严格遵循以下规则回答用户问题：
    1. 必须基于提供的参考资料（context）进行回答，不使用任何外部知识
    2. 如果参考资料中没有与问题相关的信息，直接回复"抱歉，没有找到相关内容"，不能编造答案

    参考资料:
    {context}

    用户问题:
    {question}

    """

    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )


# 检索相似文本块：从向量库查询相关内容
def retrieve_relevant_docs(
        vectorstore: Chroma,
        query: str,
        k: int = 5
) -> tuple[list, str]:
    """
    从向量库检索与查询相关的文本块

    Args:
        vectorstore: 向量数据库实例
        query: 用户查询问题
        k: 返回 top-k 个相似文本块
    Returns:
        （相关文本块列表，拼接后的上下文字符串）
    """
    # 生成查询向量并检索
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        print("未检索到相关文档")
        return [], ""

    # 拼接所有相关文本块为上下文
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print(f"检索到 {len(relevant_docs)} 个相关文本块")
    return relevant_docs, context


# ===================== 重排模型初始化 =====================
def init_rerank_model(model_name_or_path: str = "BAAI/bge-reranker-large") -> CrossEncoder:
    """
    初始化重排模型（CrossEncoder），用于对检索结果进行语义重排序.默认使用BAAI的bge-reranker-large（中文效果好）,
        可选模型：BAAI/bge-reranker-base, BAAI/bge-reranker-small

    Args:
        model_name_or_path: 重排模型名称（HuggingFace Hub规范名）或本地存储路径，
            - 1.模型名称：本地缓存（默认~/.cache/huggingface/）无该模型时，自动从Hub下载权重/配置/分词器；缓存已存在则直接加载，无需重复下载。
            - 2.本地路径：需手动下载完整模型文件（包含config.json、model.safetensors/pytorch_model.bin等）到本地路径地址。

    Returns:
        CrossEncoder: 初始化后的重排模型实例
    """
    try:
        # 加载重排模型（自动下载或使用本地缓存）
        rerank_model = CrossEncoder(model_name_or_path)
        print(f"成功加载重排模型: {model_name_or_path}")
        return rerank_model
    except Exception as e:
        raise RuntimeError(f"加载重排模型失败: {str(e)}")


# ===================== 对检索到的文档进行重排序 =====================
def rerank_documents(
        query: str,
        documents: List[Document],
        rerank_model: CrossEncoder,
        top_n: int = 3,
        score_threshold: float = 0.0
) -> List[Document]:
    """
    对检索到的文档进行重排序

    Args:
        query: 用户查询问题
        documents: 向量检索得到的原始文档列表
        rerank_model: 重排模型实例
        top_n: 重排后保留的文档数量
        score_threshold: 分数阈值，低于该值的文档会被过滤

    Returns:
        List[Document]: 重排序后的文档列表（按相关性从高到低）
    """
    if not documents:
        return []

    # 构造模型输入：(query, doc_text) 对
    pairs = [(query, doc.page_content) for doc in documents]

    # 计算相关性分数
    scores = rerank_model.predict(pairs)

    # 将文档与分数配对并排序
    doc_score_pairs = list(zip(documents, scores))
    # 按分数降序排序
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 过滤分数阈值并截取top_n
    filtered_docs = []
    for doc, score in doc_score_pairs:
        if score >= score_threshold and len(filtered_docs) < top_n:
            # 将分数添加到文档元数据中
            doc.metadata["rerank_score"] = float(score)
            filtered_docs.append(doc)
        elif len(filtered_docs) >= top_n:
            break

    print(f"重排完成：原始{len(documents)}个文档 → 筛选后{len(filtered_docs)}个文档")
    return filtered_docs


# 执行问答：检索 + 重排 + 组合提示词 + 生成
def run_rag_query(
        vectorstore: Chroma,  # 向量数据库实例
        llm: BaseChatModel,  # LLM 实例
        question: str,  # 用户查询问题
        k: int = 10,  # 检索 top-k 个文本块
        rerank_model: CrossEncoder = None,  # 重排模型
        rerank_top_n: int = 3,  # 重排后保留数量,必须小于 k
        rerank_score_threshold: float = 0.1  # 重排分数阈值，大于该阈值才被选取

) -> dict:
    """
    先检索相关文档，重排序后，再传入 LLM 生成答案

    Returns:
        问答结果（含答案、源文档信息）
    """
    if not question.strip():
        return {"error": "查询问题不能为空", "answer": "", "sources": []}

    try:
        # 步骤1：检索查询匹配输入的相关文档
        relevant_docs, context = retrieve_relevant_docs(vectorstore, question, k)

        # 步骤2：对检索的相关文档进行重排
        if rerank_model:
            filters_docs = rerank_documents(
                query=question,
                documents=relevant_docs,
                rerank_model=rerank_model,
                top_n=rerank_top_n,
                score_threshold=rerank_score_threshold
            )
            if filters_docs: # 若未重排未筛选到文档,就自动选取前rerank_top_n个检索的相关文档
                relevant_docs=filters_docs
                print(f"重排后提取 {len(relevant_docs)} 个相关文本块")
            else:
                relevant_docs=relevant_docs[:rerank_top_n]
                print(f"重排后未筛选到文档，提取检索的 {len(relevant_docs)} 个相关文本块")

        # 拼接所有相关文本块为上下文
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"组合 {len(relevant_docs)} 个相关文本块")

        # 步骤3：组合提示词
        prompt_template = get_prompt_template()
        prompt = prompt_template.format(context=context, question=question)

        # 步骤4：调用LLM生成答案
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # 整理检索返回的源文档信息（用于溯源）
        sources = [
            {
                "file_path": doc.metadata.get("source", "未知路径"),
                "page": doc.metadata.get("page", 0),
                "preview": doc.page_content[:100] + "..."
            }
            for doc in relevant_docs
        ]

        return {
            "question": question, "answer": answer, "sources": sources, "retrieved_count": len(relevant_docs)
        }

    except Exception as e:
        error_msg = f"问答执行失败：{str(e)}"
        return {
            "question": question, "answer": "抱歉，查询过程中出现错误", "sources": [], "error": error_msg
        }


# 主函数
def run_rag_main():
    """
    通用RAG(检索增强生成)核心执行流程
    完整流程：文档加载 → 文本分块 → 向量库构建 → 检索 → 重排 → LLM生成答案

    环境依赖：

    1. 千问模型配置：这里使用千问向量模型和LLM模型，需在环境变量中配置 QWEN_API_KEY（必填）
       - 获取地址：https://help.aliyun.com/zh/model-studio/get-api-key

    2. 重排模型配置：用于对检索结果进行语义重排序.默认使用BAAI的bge-reranker-large（中文效果性价比较高）。
    重排模型加载方式二选一：
        - 方式1（自动下载）：设为模型名如"BAAI/bge-reranker-large"（国内需科学上网，建议配置HF镜像）,本地缓存（默认~/.cache/huggingface/）无该模型时，自动从Hub下载模型文件；缓存已存在则直接加载，无需重复下载。
        - 方式2（本地加载）：设为本地路径（需手动下载模型文件到指定路径）
            a.模型文件获取地址：https://modelscope.cn/models/BAAI/bge-reranker-v2-m3
            b.需下载文件：config.json、model.safetensors、special_tokens_map.json、tokenizer.json、tokenizer_config.json
    """
    doc_dir = "./docs"  # 替换为你的文档目录
    file_types = ["pdf", "docx", "txt"]  # 要加载的文件类型列表
    embedding_model = "text-embedding-v4"  # 千问(DashScope) 的文本嵌入模型
    llm_model = "qwen-plus"  # 千问(DashScope) 的大语言模型
    persist_dir = "./chroma_db"  # 向量数据库存储目录
    # rerank_model_name = "BAAI/bge-reranker-large"  # 重排模型名称，自动从huggingface下载，并缓存到~/.cache/huggingface/
    rerank_model_name = "../data/models_reranker_data/BAAI/bge-reranker-v2-m3"  # 本地模型路径，需手动下载模型到本路径
    retrieve_k = 8  # 检索 top-k 个文本块
    rerank_top_n = 5  # 重排后保留数量,必须小于 k
    rerank_score_threshold = 0.1  # 重排分数阈值，大于该阈值才被选取

    # 1. 加载文档
    docs = load_documents(doc_dir, file_types)
    if not docs:
        return

    # 2. 文本分块
    split_docs = split_documents(docs, chunk_size=1000, chunk_overlap=200)

    # 3. 分块文本向量化并持久化到向量数据库中
    vectorstore = create_and_persist_vectorstore(split_docs, embedding_model, persist_dir)

    # 4. 初始化 LLM
    llm = langchain_qwen_llm(llm_model, temperature=0)

    # 5. 初始化重排模型
    reranker_model = init_rerank_model(rerank_model_name)

    # 测试问答
    test_questions = [
        "文档的核心讲的是什么？",
        "公司的发展历程？",
    ]

    for question in test_questions:
        print(f"\n\n问题: {question}")
        print("=" * 50)
        # 6. 基于检索内容生成答案(检索 + 重排 + 组合提示词 + 生成)
        response = run_rag_query(
            vectorstore=vectorstore,
            llm=llm,
            question=question,
            k=retrieve_k,
            rerank_model=reranker_model,
            rerank_top_n=rerank_top_n,
            rerank_score_threshold=rerank_score_threshold
        )
        print(f"回答: \n{response['answer']}")
        print("来源文档:",
              [f"{doc.get('file_path', '未知')},第{doc.get('page', '未知')}页" for doc in response['sources']])


if __name__ == "__main__":
    run_rag_main()
