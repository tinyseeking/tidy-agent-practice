# 安装 ColBERT  pip install colbert-ai
# 额外依赖（适配模型加载） pip install torch transformers faiss-cpu  # GPU版可装 faiss-gpu
from typing import List, Tuple, Optional, Union
import os
import torch
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig


class ColBERTReranker:
    """
    ColBERT v2 重排器（适配百万级候选集的高效检索重排）
    核心特性：
    - 预编码构建量化索引，支持百万级段落高效检索
    - 后期交互机制，兼顾精度与检索速度
    """
    # 默认配置（平衡精度/速度/存储）
    DEFAULT_CHECKPOINT = "colbert-ir/colbertv2.0"
    DEFAULT_TOP_K = 100

    def __init__(
            self,
            checkpoint: str = DEFAULT_CHECKPOINT,
            index_root: str = "./data/colbert_index",
            nbits: int = 2,  # 2bit量化，百万级数据仅占GB级存储
            doc_maxlen: int = 512,
            nranks: int = 1,
            experiment_name: str = "default",
            device=None
    ):
        """
        初始化 ColBERT 重排器

        Args:
            checkpoint: ColBERT 模型权重（设置为HuggingFace模型名称/本地路径）
                - 1.若为模型名称：如果本地缓存（默认路径~ /.cache/huggingface/）中没有该模型，会自动从 Hugging Face Hub 下载模型权重、配置和分词器文件。
                若本地已存在该模型，代码会直接加载缓存中的文件，无需重新下载。由于国内无法访问huggingface，需要设置代理或镜像源方式。
                - 2.若为本地路径：则需要手动下载模型文件到路径地址.
            index_root: 索引存储根路径
            nbits: 量化位数（1/2/4/8，位数越低存储越小、精度略降）
            doc_maxlen: 文档最大长度
            nranks: 并行进程数（单机设1，分布式可增大）
            experiment_name: 实验名称（用于ColBERT日志）
        """
        # 基础配置
        self.checkpoint = checkpoint
        self.index_root = os.path.abspath(index_root)  # 索引根路径
        self.experiment_name = experiment_name
        self.nbits = nbits
        self.doc_maxlen = doc_maxlen
        self.nranks = nranks
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # 核心组件
        self.indexer: Optional[Indexer] = None
        self.searcher: Optional[Searcher] = None
        self.collection: Optional[List[str]] = None  # 存储原始段落集合（索引构建后缓存）
        # 日志提示
        print(f"🔧 ColBERT 初始化完成 | 模型：{checkpoint} | 索引路径：{self.index_root}")

    def _get_colbert_config(self) -> ColBERTConfig:
        """构建 ColBERT 配置对象（封装通用配置逻辑）"""
        config = ColBERTConfig(
            nbits=self.nbits,
            root=self.index_root,
            experiment=self.experiment_name,  # 关联实验名称
            doc_maxlen=self.doc_maxlen,
            bsize=128,  # 批量编码大小,额外优化参数（百万级场景）
        )
        return config

    def build_index(
            self,
            passages: List[str],
            index_name: str = "english_kb",
            overwrite: bool = False
    ) -> Indexer | None:
        """
        构建 ColBERT 量化索引（支持百万级段落）

        Args:
            passages: 段落列表（支持百万级数据）
            index_name: 索引名称（存储在index_path下的子目录）
            overwrite: 是否覆盖已有索引

        Returns:
            初始化后的 Indexer 实例

        Raises:
            ValueError: 段落列表为空或索引已存在且未开启overwrite
            RuntimeError: 索引构建失败
        """
        # 参数校验
        if not passages:
            raise ValueError("段落列表不能为空，无法构建索引")

        # 索引路径校验，Indexer默认使用的索引路径由root（实验根路径）/experiment（实验名）/indexes/index_name（索引名）拼接
        full_index_path = os.path.join(self.index_root, self.experiment_name, "indexes", index_name)
        if os.path.exists(full_index_path) and not overwrite:
            print(f"索引已存在：{full_index_path} | 如需覆盖请设置 overwrite=True")
            return

        # 缓存原始段落（用于检索时映射ID到文本）
        self.collection = passages

        try:
            print(f"🚀 开始构建 ColBERT 索引 | 段落数量：{len(passages):,} | 索引名称：{index_name}")
            run_config = RunConfig(
                nranks=self.nranks,
                root=self.index_root,  # 实验根路径
                experiment=self.experiment_name,  # 实验名
            )
            with Run().context(run_config):
                _config = self._get_colbert_config()
                self.indexer = Indexer(checkpoint=self.checkpoint, config=_config)
                # 构建索引（自动分批处理百万级数据）
                index_path = self.indexer.index(
                    name=index_name,
                    collection=passages,
                    overwrite=overwrite
                )

            print(f"✅ ColBERT 索引构建完成 | 存储路径：{index_path}")
            return self.indexer
        except Exception as e:
            raise RuntimeError(f"ColBERT 索引构建失败：{str(e)}") from e

    def load_index(self, index_name: str = "english_kb", collection: Optional[List[str]] = None) -> Searcher:
        """
        加载已构建的索引（避免重复构建，提升复用效率）

        Args:
            index_name: 索引名称
            collection: 原始段落列表（若未缓存需传入，用于映射ID到文本）

        Returns:
            初始化后的 Searcher 实例

        Raises:
            FileNotFoundError: 索引路径不存在
        """
        full_index_path = os.path.join(self.index_root, self.experiment_name, "indexes", index_name)

        if not os.path.exists(full_index_path):
            raise FileNotFoundError(f"索引不存在：{full_index_path} | 请先调用 build_index 构建索引")

        try:
            print(f"📥 加载 ColBERT 索引 | 路径：{full_index_path}")
            # RunConfig 关联路径和实验名
            run_config = RunConfig(
                nranks=self.nranks,
                root=self.index_root,
                experiment=self.experiment_name,
            )
            with Run().context(run_config):
                config = self._get_colbert_config()
                self.searcher = Searcher(
                    index=index_name,
                    checkpoint=self.checkpoint,
                    config=config
                )
            # 关联原始集合
            if collection is not None:
                self.searcher.collection = collection
            print(f"✅ ColBERT 索引加载完成 | 索引规模：{len(self.searcher.collection):,} 段落")
            return self.searcher
        except Exception as e:
            raise RuntimeError(f"ColBERT 索引加载失败：{str(e)}") from e

    def rerank(
            self,
            query: str,
            top_k: int = DEFAULT_TOP_K,
            index_name: str = "english_kb",
            load_index_if_needed: bool = True,
            collection: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        基于 ColBERT 索引的高效重排

        Args:
            query: 查询语句（ColBERT v2 推荐英文，中文需适配中文版模型）
            top_k: 返回前k条最相关段落
            index_name: 索引名称
            load_index_if_needed: 若未加载索引，是否自动加载
            collection: 原始段落列表（若未缓存需传入，用于映射ID到文本）

        Returns:
            排序结果列表：[(段落文本, 相关性得分), ...]（按得分降序）

        Raises:
            RuntimeError: 索引未加载且自动加载失败

        """
        # 参数校验
        if top_k <= 0:
            raise ValueError(f"top_k 必须大于0，当前值：{top_k}")
        if not query.strip():
            raise ValueError("查询语句不能为空")

        # 自动加载索引
        if self.searcher is None and load_index_if_needed:
            self.load_index(index_name=index_name,collection=collection)

        # 校验索引是否就绪
        if self.searcher is None:
            raise RuntimeError("索引未加载 | 请先调用 build_index 或 load_index")

        try:
            print(f"开始 ColBERT 重排 | 查询：{query[:50]}... | Top-K：{top_k}")
            with Run().context(RunConfig(nranks=self.nranks)):
                # 核心检索（预编码+后期交互，百万级低延迟）
                results = self.searcher.search(query, k=top_k)
                # 解析结果：(段落ID, 得分) → (段落文本, 得分)
                pid_list, _, score_list = results  # 拆分3个列表（忽略第二个辅助列表）
                ranked_results = []
                for pid, score in zip(pid_list, score_list):  # 按位置一一对应
                    try:
                        passage_text = self.searcher.collection[pid]
                        ranked_results.append((passage_text, float(score)))
                    except KeyError:
                        print(f"警告：pid {pid} 在collection中不存在，跳过")

            print(f"ColBERT 重排完成 | 返回结果数：{len(ranked_results)}")
            return ranked_results
        except Exception as e:
            raise RuntimeError(f"ColBERT 重排失败：{str(e)}") from e

    def release(self):
        """释放资源"""
        if self.indexer is not None:
            del self.indexer
        if self.searcher is not None:
            del self.searcher
        self.collection = None
        print("✅ ColBERT 资源已释放")


# -------------------------- 测试示例 --------------------------
if __name__ == "__main__":
    # 1. 初始化 ColBERT 重排器
    colbert_reranker = ColBERTReranker(
        # 模型权重路径（二选一）：
        # 1. 填写HuggingFace模型名称，默认从HuggingFace自动加载,如：checkpoint="colbert-ir/colbertv2.0"
        # 2. 填写模型本地存储路径：需提前下载模型文件到指定路径
        #    模型文件获取地址：https://modelscope.cn/models/colbert-ir/colbertv2.0/
        #    需下载文件：config.json、model.safetensors、special_tokens_map.json、tokenizer.json、tokenizer_config.json、vocab.txt
        checkpoint="../../data/models_reranker_data/colbert-ir/colbertv2",  # 本地路径
        # checkpoint="./colbertv2_models",  # 本地路径
        index_root="./colbert_index",
        # 可选扩展配置（按需启用）：
        # nbits=2,                # 量化位数（1/2/4/8，位数越低存储越小、精度略降）
        # doc_maxlen=512,         # 文档最大token长度（适配BERT类模型）
        # experiment_name="colbert_v2_rerank",  # 实验名（用于索引路径分级）
        # device="cuda",          # 运行设备（自动优先GPU，无则用CPU）
        # nranks=1,               # 并行进程数（单机建议设1，分布式可增大）
    )

    # 2. 模拟百万级段落（实际可替换为真实数据）
    test_passages = [f"Document {i}: RAG optimization for large-scale knowledge base {i}" for i in range(1000)]
    # 插入目标段落（高相关）
    test_passages[100] = "Document 100: ColBERT v2 achieves 10x throughput for million-scale reranking"
    test_passages[200] = "Document 200: ColBERT v2 is optimized for million-scale reranking with 2-bit quantization"
    test_passages[400] = "Document 400: Memory-efficient index for 1M+ passages using ColBERT v2 2-bit quantization"
    # 插入无关段落（测试重排精度）
    test_passages[150] = "Document 150: Weather forecast for New York on day 2 - sunny with 25°C"
    test_passages[250] = "Document 250: How to bake a cake - step by step guide for beginners"
    test_passages[350] = "Document 350: Car maintenance tips for gasoline engines 3"

    # 3. 构建索引（首次运行耗时，后续可复用）
    try:
        colbert_reranker.build_index(
            passages=test_passages,
            index_name="english_kb",
            overwrite=False  # 覆盖已有索引（测试用）
        )
    except RuntimeError as e:
        print(f"索引构建失败：{e}")
        exit(0)

    # 4. 执行重排
    query = "Which model is efficient for million-scale English reranking?"
    try:
        results = colbert_reranker.rerank(
            query=query,
            top_k=5,
            index_name="english_kb",
            collection=test_passages
        )
    except RuntimeError as e:
        raise RuntimeError(f"重排失败：{str(e)}") from e

    print("\n===== ColBERT v2 百万级候选重排结果（前5条）=====")
    for i, (passage, score) in enumerate(results, 1):
        print(f"\n{i}. 得分：{score:.4f} 文档：{passage[:80]}")

    # 5. 释放资源
    colbert_reranker.release()
