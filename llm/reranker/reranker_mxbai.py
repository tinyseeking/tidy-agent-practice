import logging
from typing import List, Tuple, Optional, Dict, Any

import torch
from sentence_transformers import CrossEncoder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MxbaiReranker:
    """
    MixedBread AI Rerank 模型封装类
    支持多语言（德/英/西/法/中等）的文本重排
    """

    def __init__(
            self,
            model_name_or_path: str = "mixedbread-ai/mxbai-rerank-large-v2",
            max_length: int = 8192,
            device: Optional[str] = None,
            model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        初始化重排模型

        Args:
            model_name_or_path: 模型名称或本地路径
                -若为模型名称：如果本地缓存（默认路径~ /.cache/huggingface/）中没有该模型，会自动从 Hugging Face Hub 下载模型权重、配置和分词器文件。
                如果本地缓存已存在该模型，代码会直接加载缓存中的文件，无需重新下载。由于国内无法访问huggingface，需要设置代理或镜像源方式。
                -若为本地路径：则需要手动下载模型文件到路径地址（config.json、model.safetensors(或pytorch_model.bin)、
                tokenizer.json、tokenizer_config.json、special_tokens_map.json、vocab.txt）.
            max_length: 文本最大长度
            device: 运行设备 (e.g., "cpu", "cuda", "cuda:0")
            model_kwargs: 传递给CrossEncoder的额外参数
        """
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        # 设备自动检测
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_kwargs = model_kwargs or {}

        # 加载模型
        self._load_model()

    def _load_model(self) -> None:
        """加载CrossEncoder模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name_or_path}")
            self.model = CrossEncoder(
                model_name_or_path=self.model_name_or_path,
                max_length=self.max_length,
                device=self.device,
                **self.model_kwargs
            )
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def rerank_sorted(
            self,
            query: str,
            passages: List[str],
            batch_size: int = 16,
            return_scores: bool = True
    ) -> List[Tuple[str, float]] | List[str]:
        """
        对段落进行重排

        Args:
            query: 多语言查询文本
            passages: 待重排的段落列表
            batch_size: 批量推理大小,可根据CPU/GPU调整，CPU建议设为8/16
            return_scores: 是否返回得分

        Returns:
            排序后的段落列表（可选包含得分）
        """
        if not query:
            raise ValueError("查询文本不能为空")
        if not passages:
            raise ValueError("段落列表不能为空")

        try:
            # 构造输入对
            inputs = [[query, passage] for passage in passages]

            # 批量推理
            logger.debug(f"开始推理，批次大小: {batch_size}")
            # 推理打分（batch_size可根据CPU/GPU调整，CPU建议设为8/16）
            scores = self.model.predict(
                inputs,
                batch_size=batch_size,
                show_progress_bar=False
            )

            # 排序
            ranked_pairs = sorted(
                zip(passages, scores),
                key=lambda x: x[1],
                reverse=True
            )

            logger.info(f"重排完成，共处理 {len(passages)} 个段落")

            if return_scores:
                return ranked_pairs
            else:
                return [pair[0] for pair in ranked_pairs]

        except Exception as e:
            logger.error(f"重排失败: {e}")
            raise

    def release(self):
        """显式释放模型资源（建议手动调用）"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("模型资源已释放")
        except Exception as e:
            logger.warning(f"释放模型资源时出错: {e}")


# 使用示例
if __name__ == "__main__":

    # 初始化重排器
    reranker = MxbaiReranker(
        # 模型名称或本地路径
        # 1.若为模型名称：如果本地缓存（默认路径~ /.cache/huggingface/）无该模型时，会自动从 Hugging Face Hub 下载模型权重、配置和分词器文件。
        #     若已存在该模型，代码会直接加载缓存中的文件，无需重新下载。
        # 2.若为本地路径：则需要手动下载模型文件到路径地址（config.json、model.safetensors(或pytorch_model.bin)、
        #     tokenizer.json、tokenizer_config.json、special_tokens_map.json、vocab.json）.
        # 这里使用本地模型路径，从huggingface.co 或（国内）modelscope.cn中下载模型(mxbai-rerank-large-v2)，存放到本地路径。
        # model_name_or_path="mixedbread-ai/mxbai-rerank-large-v2", # 模型名称,自动下载到~/.cache/huggingface/
        model_name_or_path="../../data/models_reranker_data/mixedbread-ai/mxbai-rerank-large-v2",  # 本地模型路径
        max_length=8192,
        # device="cuda"  # 如果有GPU可以指定
    )

    # 测试数据
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

    # 执行重排
    results = reranker.rerank_sorted(query, docs, batch_size=8)

    # 打印结果
    print(f"\nmxbai-rerank-large-v2 重排结果：")
    print(f"query:{query}")
    print("-" * 80)
    for i, (passage, score) in enumerate(results, 1):
        print(f"第{i}名：得分 {score:.4f} | 内容：{passage}")
