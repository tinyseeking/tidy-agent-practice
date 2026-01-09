import os
from typing import List, Tuple, Optional

import torch
from FlagEmbedding import FlagReranker


class RerankerBgeV2:
    """
    BGE-Reranker-V2 重排模型处理器

    基于BAAI推出的bge-reranker-v2-m3模型实现文本相关性重排，用于提升检索系统中
    召回阶段候选文档的排序精度，核心通过计算query-doc对的语义相似度分数实现重排。
    适用于中文/多语言场景的检索后重排任务。
    """

    def __init__(
            self,
            model_path: str = "../../data/models_reranker_data/BAAI/bge-reranker-v2-m3",
            use_fp16: bool = True,
            device: str = "cpu"
    ):
        """
        初始化重排模型
            model_path: 模型本地存储路径，需包含完整的模型文件（config.json、model.safetensors等）
            use_fp16: 是否启用FP16精度推理，GPU环境下推荐True，CPU环境会自动强制关闭
            device: 模型运行设备，支持"cpu"、"cuda"、"cuda:0"等格式，为空时自动检测可用设备
        """
        # 转换为绝对路径，避免相对路径导致的文件查找失败
        self.model_path = os.path.abspath(model_path)
        # 设备自动检测
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # CPU环境下FP16无性能收益且可能导致兼容性问题，强制关闭
        self.use_fp16 = use_fp16 if device != "cpu" else False
        # 重排模型实例，延迟初始化
        self.reranker: Optional[FlagReranker] = None

        # 初始化模型
        self._init_model()

    def _init_model(self) -> None:
        """加载并重初始化重排模型"""
        try:
            self.reranker = FlagReranker(
                model_name_or_path=self.model_path,
                use_fp16=self.use_fp16,
                device=self.device
            )
            print(f"✅ 重排模型加载成功，设备：{self.device}，FP16：{self.use_fp16}")
        except Exception as e:
            raise RuntimeError(f"模型初始化失败：{e}")

    def rerank_sorted(
            self,
            query: str,
            candidate_docs: List[str],
            top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        对候选文档列表按与查询语句的相关性进行重排

        Args:
            query: 用户查询语句（文本）
            candidate_docs: 待重排的候选文档列表
            top_k: 返回前k个结果（None返回全部）
        Returns:
           重排结果列表，元素为(文档文本, 相关性分数)，按分数降序排列
        """
        if not candidate_docs:
            raise ValueError("候选文档列表不能为空")

        if self.reranker is None:
            raise RuntimeError("模型未初始化，请检查模型路径和设备配置")

        # 构造query-doc文本对,格式：[[query1, doc1], [query1, doc2], ...]
        text_pairs = [[query, doc] for doc in candidate_docs]

        # 批量计算相关性分数
        try:
            scores = self.reranker.compute_score(text_pairs)
        except Exception as e:
            raise RuntimeError(f"分数计算失败：{e}")

        # 按分数降序排序
        ranked_results = sorted(
            zip(candidate_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # 截取top_k结果
        if top_k is not None and top_k > 0:
            ranked_results = ranked_results[:top_k]

        return ranked_results


# 示例：BGE-Reranker-V2模型的完整使用流程
if __name__ == "__main__":

    # 1. 初始化重排处理器
    # 下载模型文件到存储路径../../data/models_reranker_data/BAAI，模型下载地址：
    # - HuggingFace: https://huggingface.co/BAAI/bge-reranker-v2-m3
    # - ModelScope: https://www.modelscope.cn/models/BAAI/bge-reranker-v2-m3
    # 需下载的核心文件：config.json、model.safetensors(或pytorch_model.bin)、 tokenizer.json、
    #  tokenizer_config.json、special_tokens_map.json、vocab.json。
    reranker = RerankerBgeV2(
        model_path="../../data/models_reranker_data/BAAI/bge-reranker-v2-m3",
        # device="cuda:0",  # 如需使用GPU可显式指定
        # use_fp16=True     # GPU环境下建议开启FP16
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
    results = reranker.rerank_sorted(
        query=query,
        candidate_docs=docs,
        top_k=None  # 可改为None返回全部
    )

    # 4. 打印结果
    print("\n" + "=" * 50)
    print(f"bge-reranker-v2-m3 重排结果（分数越高越相关）：")
    print(f"查询语句：{query}")
    print("=" * 50)
    for idx, (doc, score) in enumerate(results, 1):
        print(f"Top{idx} | 分数：{score:.4f} | 文档：{doc}")
    print("=" * 50 + "\n")
