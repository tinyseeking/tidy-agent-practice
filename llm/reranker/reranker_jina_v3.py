from typing import List, Tuple, Optional, Union
import torch
from transformers import AutoModel


class JinaRerankV3:
    """
    基于 transformers.AutoModel 封装的 jina-rerank-v3 重排器
    """
    DEFAULT_MODEL_NAME = "jinaai/jina-rerank-v3"

    def __init__(
            self,
            model_name_or_path: str = DEFAULT_MODEL_NAME,
            device: Optional[str] = None,
            batch_size: int = 32,  # jina-rerank-v3 批量推理最优批次
            trust_remote_code: bool = True,
            dtype: Union[str, torch.dtype] = "auto"
    ):
        """
        初始化 jina-rerank-v3 重排器（基于 AutoModel 原生接口）

        Args:
            model_name_or_path: 模型本地路径或HuggingFace名称
            device: 运行设备（auto/cuda/cpu）
            batch_size: 推理批次大小
            trust_remote_code: 是否允许加载模型仓库中自定义的代码（如 modeling_*.py、等），并执行这些代码来初始化模型/配置。
            dtype: 模型数据类型（auto/float16/float32）
        """
        # 设备适配
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype if isinstance(dtype, torch.dtype) else dtype

        # 模型初始化
        self.model: Optional[AutoModel] = None
        self._load_model()

    def _load_model(self) -> None:
        """重载模型加载方法：使用 AutoModel 加载 jina-rerank-v3"""
        try:
            # 加载原生 AutoModel（jina官方推荐方式）
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                dtype=self.dtype,
                device_map=self.device  # 自动分配设备
            )
            # 设置评估模式
            self.model.eval()
            print(
                f"✅ Jina-Rerank-V3 模型加载完成 | 设备：{self.device} | 批次大小：{self.batch_size} | "
                f"数据类型：{self.dtype} | 评估模式：{self.model.training}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Jina-Rerank-V3 模型加载失败：{e}\n"
                "排查建议：\n"
                "1. 模型文件是否完整 \n"
                "2. trust_remote_code 是否设为True \n"
                "3. transformers版本是否≥4.36.0"
            )

    def rerank_sorted(
            self,
            query: str,
            candidate_docs: List[str],
            return_scores: bool = True,
            top_k: Optional[int] = None
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        核心重排方法
        调用 jina-rerank-v3 原生 rerank 方法，返回排序后的结果

        Args:
            query: 查询语句（支持多语言）
            candidate_docs: 候选文档列表
            return_scores: 是否返回相关性得分
            top_k: 返回前k个结果（None返回全部）

        Returns:
            排序后的文档列表 或 (文档, 得分) 元组列表
        """
        # 输入合法性校验
        if not candidate_docs:
            raise ValueError("候选文档列表不能为空")
        if self.model is None:
            raise RuntimeError("模型未初始化，请检查模型加载是否成功")

        # 核心推理：调用 jina 原生 rerank 方法
        try:
            with torch.no_grad():  # 禁用梯度计算（节省显存）
                results = self.model.rerank(
                    query=query,
                    documents=candidate_docs,
                    top_n=top_k if top_k else len(candidate_docs)
                )
        except Exception as e:
            raise RuntimeError(f"重排推理失败：{e}")

        # 解析结果（按得分降序排列，jina原生结果已排序）
        ranked_results = [
            (result["document"], float(result["relevance_score"]))
            for result in results
        ]

        # 适配返回格式
        if return_scores:
            return ranked_results
        else:
            return [item[0] for item in ranked_results]

    def release(self) -> None:
        """资源释放：清理模型和GPU缓存"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🔌 Jina-Rerank-V3 模型资源已释放")


# -------------------------- 测试示例 --------------------------
if __name__ == "__main__":
    # 1. 初始化重排器（本地模型路径）
    reranker = JinaRerankV3(
        model_name_or_path="../../data/models_reranker_data/jinaai/jina-reranker-v3",
        batch_size=16
    )

    # 2. 测试数据（多语言混合）
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

    # 3. 重排结果
    print("===== 重排结果=====")
    results_with_scores = reranker.rerank_sorted(
        query=query,
        candidate_docs=docs,
        return_scores=True
    )
    for idx, (doc, score) in enumerate(results_with_scores, 1):
        print(f"{idx}. 得分：{score:.4f} 文档：{doc[:100]}")

    # 4. 释放资源
    reranker.release()
