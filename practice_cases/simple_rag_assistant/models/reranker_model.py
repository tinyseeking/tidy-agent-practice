import logging
from typing import List, Optional

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RerankerCrossModel:
    def __init__(
            self,
            model_name_or_path: str = "BAAI/bge-reranker-large",
            device: Optional[str] = None,
            batch_size: int = 16
    ):
        """
        初始化重排器。用于对检索阶段召回的候选文档进行语义相关性重排，提升检索精度。
        适配遵循 sentence-transformers 的 CrossEncoder 规范的模型, 如ms-marco-MiniLM-L-12-v2、bge-reranker-v2-m3/bge-reranker-large等

        Args:
            model_name_or_path: str，模型名称（HuggingFace Hub规范名）或本地存储路径：
                - 1.模型名称：本地缓存（默认~/.cache/huggingface/）无该模型时，自动从Hub下载权重/配置/分词器；缓存已存在则直接加载，无需重复下载。
                - 2.本地路径：需手动下载完整模型文件（包含config.json、model.safetensors/pytorch_model.bin等）到本地路径地址。
            device: 模型运行设备，None则自动检测（优先使用CUDA，无则使用CPU）
            batch_size: 推理批次大小，建议CPU设8/16，GPU可根据显存适当增大（默认16）
        """
        # 设备自动适配
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_name_or_path = model_name_or_path
        self.reranker_model: Optional[CrossEncoder] = None

        # 初始化模型
        self._load_model()

    def _load_model(self) -> None:
        """加载并重初始化CrossEncoder重排模型"""
        try:
            self.reranker_model = CrossEncoder(self.model_name_or_path, device=self.device)
            logger.info(f"✅重排模型加载完成 | 设备：{self.device} | 批次大小：{self.batch_size}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查：1. 模型路径是否正确 2. 网络是否正常（首次下载需联网）")

    def rerank_documents(
            self,
            query: str,
            documents: List[Document],
            top_n: int = 3,
            score_threshold: float = 0.0
    ) -> List[Document]:
        """
        对检索到的文档进行重排序

        Args:
            query: 用户查询问题
            documents: 向量检索得到的原始文档列表
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
        scores = self.reranker_model.predict(pairs)

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

        logger.info(f"重排完成：原始{len(documents)}个文档 → 筛选后{len(filtered_docs)}个文档")
        return filtered_docs


if __name__ == "__main__":
    # 测试数据
    sample_documents = [
        Document(page_content="文档1内容：人工智能入门", metadata={"id": 1}),
        Document(page_content="文档2内容：大语言模型原理", metadata={"id": 2}),
        Document(page_content="文档3内容：Python 基础教程", metadata={"id": 3}),
        Document(page_content="文档4内容：语义检索算法", metadata={"id": 4}),
        Document(page_content="文档5内容：机器学习实战", metadata={"id": 5}),
    ]
    # 使用BAAI/bge-reranker-large模型
    reranker = RerankerCrossModel(
        # 这里加载本地路径模型（需手动下载模型文件到指定路径）
        # a.模型文件获取地址：https://modelscope.cn/models/BAAI/bge-reranker-v2-m3
        # b.需下载文件：config.json、model.safetensors、special_tokens_map.json、tokenizer.json、tokenizer_config.json
        # model_name_or_path="BAAI/bge-reranker-base",  # 模型名称
        model_name_or_path="../../../data/models_reranker_data/BAAI/bge-reranker-v2-m3",  # 模型名称
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8
    )
    query = "大语言模型的语义检索方法"
    result_docs = reranker.rerank_documents(
        query=query,
        documents=sample_documents,
        top_n=3
    )

    # 打印最终结果（新增：输出重排后的详细信息）
    print("\n========== 重排结果详情 ==========")
    for i, doc in enumerate(result_docs, 1):
        print(f"\n第{i}名文档：")
        print(f"文档ID：{doc.metadata['id']}  重排分数：{doc.metadata['rerank_score']:.4f} 文档内容：{doc.page_content}")
