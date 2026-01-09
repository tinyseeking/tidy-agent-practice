from typing import List, Tuple, Optional, Union

import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
            self,
            model_name_or_path: str = "../data/models_data/cross-encoder/ms-marco-MiniLM-L6-v2",
            device: Optional[str] = None,
            batch_size: int = 16
    ):
        """
        初始化重排器。
        基于CrossEncoder的文档重排器，用于检索系统中对召回阶段的候选文档进行语义相关性重排，提升检索精度。
        适配遵循 sentence-transformers 的 CrossEncoder 规范的模型, 如ms-marco-MiniLM-L-12-v2、bge-reranker-v2-m3等

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
        self.model: Optional[CrossEncoder] = None

        # 初始化模型
        self._load_model()

    def _load_model(self) -> None:
        """加载并重初始化CrossEncoder重排模型"""
        try:
            self.model = CrossEncoder(self.model_name_or_path, device=self.device)
            print(f"✅重排模型加载完成 | 设备：{self.device} | 批次大小：{self.batch_size}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败：{e}\n请检查：1. 模型路径是否正确 2. 网络是否正常（首次下载需联网）")

    def rerank_sorted(
            self,
            query: str,
            candidate_docs: List[str],
            return_scores: bool = True,
            top_k: Optional[int] = None
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        对候选文档进行相关性重排

        Args:
            query: 用户查询语句
            candidate_docs: 召回阶段的候选文档列表（建议top-20以内，保证重排效率）
            return_scores: 是否返回相关性得分（得分越高，相关性越强）
            top_k: 返回前k个最相关文档，None返回全部

        Returns:
            重排后的文档列表（或文档+得分的元组列表），按相关性降序排列

        Raises:
            ValueError: 候选文档列表为空时抛出
            RuntimeError: 模型未成功加载时抛出
        """
        # 输入合法性校验
        if not candidate_docs:
            raise ValueError("候选文档列表不能为空")

        if self.model is None:
            raise RuntimeError("模型未初始化，请检查模型加载是否成功")

        # 构造query-doc配对（CrossEncoder必需输入格式）
        doc_pairs = [[query, doc] for doc in candidate_docs]

        # 批量推理计算相关性得分
        try:
            scores = self.model.predict(doc_pairs, batch_size=self.batch_size)
        except Exception as e:
            raise RuntimeError(f"重排打分失败：{e}")

        # 按得分降序排序
        sorted_indices = scores.argsort()[::-1]
        ranked_results = [
            (candidate_docs[i], float(scores[i])) for i in sorted_indices
        ]

        # 截取top_k结果
        if top_k is not None and top_k > 0:
            ranked_results = ranked_results[:top_k]

        # 适配返回格式
        if return_scores:
            return ranked_results
        else:
            return [item[0] for item in ranked_results]

    def release(self):
        """显式释放模型资源（建议手动调用）"""
        if self.model is not None:
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🔌 重排模型资源已释放")


# 示例使用
if __name__ == "__main__":
    # 1. 创建重排器实例
    reranker = CrossEncoderReranker(
        # model_name_or_path="cross-encoder/ms-marco-MiniLM-L6-v2",  # 使用HuggingFace远程加载模型
        # 手动从huggingface.co或modelscope.cn中下载模型文件，存储本地路径下
        model_name_or_path="../../data/models_reranker_data/cross-encoder/ms-marco-MiniLM-L6-v2",
        # device="cpu",  # 强制使用CPU
        batch_size=16
    )

    # 2. 准备测试数据
    query = "人工智能在医疗领域的应用"
    docs = [
        "人工智能驱动的农业无人机可实现作物病虫害监测，精准率达85%以上-E",  # 无关（完全跨领域）
        "AI辅助诊断系统可快速分析CT、MRI等医学影像，使肺癌、乳腺癌等疾病的早期检出率提升30%以上-A",  # 高相关
        "深度学习算法在金融风控中的应用案例，有效降低信贷违约率15%-E",  # 无关（金融领域）
        "卷积神经网络在皮肤癌图像分类任务上达到91%的准确率，为基层医疗机构提供了低成本诊断方案-B",  # 中高相关（技术细分，间接相关）
        "智能模拟在生物医药研发中的应用，主要用于药物分子结构模拟，不属于临床医疗应用范畴-D",  # 干扰项（语义相近但领域不符）
        "人工智能技术在医疗领域的核心应用包括AI辅助诊断、个性化治疗方案生成、医疗影像分析等-A",  # 高相关
        "机器学习算法在医保风控系统中的应用，可识别虚假就医报销行为，属于AI在医疗管理的边缘场景-D",  # 低相关（跨领域AI，弱关联）
        "医疗大数据平台通过机器学习算法整合患者电子病历，为医院管理决策提供数据支撑-C",  # 中相关（泛医疗AI，关联性减弱）
    ]

    # 3. 执行重排
    # 返回带得分的结果
    ranked_docs_with_scores = reranker.rerank_sorted(
        query=query,
        candidate_docs=docs,
        return_scores=True,
        top_k=None  # 可指定返回前N个，如top_k=2
    )

    # 4. 输出结果
    print("\n=== ms-marco-MiniLM-L6-v2 重排结果===")
    print(f"query:{query}")
    for idx, (doc, score) in enumerate(ranked_docs_with_scores, 1):
        print(f"TOP-{idx} | 得分：{score:.4f}  文档：{doc}")
