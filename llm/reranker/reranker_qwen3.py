# 安装依赖：pip install transformers>=4.51.0 torch modelscope
import torch
from typing import List, Optional, Tuple
from modelscope import AutoTokenizer, AutoModelForCausalLM


class QwenReranker:
    """
    通义千问重排模型封装类

    基于ModelScope的Qwen3-Reranker系列模型，实现查询(Query)与文档(Document)的相关性评分及排序，
    模型通过判断文档是否满足查询需求，输出0-1之间的相关性得分（越接近1表示相关性越高）。

    基础用法示例：
        >>> ranker = QwenReranker("Qwen/Qwen3-Reranker-0.6B")
        >>> query = "人工智能在医疗领域的应用"
        >>> docs = ["AI辅助诊断系统可分析医学影像...", "人工智能在农业中的应用..."]
        >>> sorted_results = ranker.rerank_sorted(query, docs)
        >>> print(sorted_results)  # [(doc索引, 得分), ...]（按得分降序）
    """

    DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(
            self,
            model_name_or_path: str = "Qwen/Qwen3-Reranker-0.6B",
            device: Optional[str] = None,
            max_length: int = 8192,
    ):
        """
        初始化重排模型

        Args:
            model_name_or_path: 模型名称或本地路径。
                1.若为模型名称：如果本地缓存（默认用户路径 ~/.cache/modelscope/hub ）中没有模型文件，则 from_pretrained() 会自动
                从modelscope.cn下载模型权重、配置和分词器文件。 如果本地缓存已存在该模型，代码会直接加载缓存中的文件，无需重新下载。
                2.若为本地模型的存储路径：则需要手动下载模型文件到路径地址（config.json、*.safetensors 等）.
            device: 运行设备（"cuda"/"cpu"），默认自动检测
            max_length: 默认最大文本长度
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        self.max_length = max_length

        # 设备自动检测
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).eval()
        # We recommend enabling flash_attention_2 for better acceleration and memory saving.
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2").cuda().eval()

        if device and device != "auto":  # 指定单卡
            self.model = self.model.to(device)

        # 特殊 token id
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        # prompt 模板
        self.prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
            "Instruct provided. Note that the answer can only be \"yes\" or "
            "\"no\".<|im_end|>\n<|im_start|>user\n")
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    @staticmethod
    def _format(instruction: Optional[str], query: str, doc: str) -> str:
        inst = instruction or QwenReranker.DEFAULT_INSTRUCTION
        return f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {doc}"

    def _tokenize(self, texts: List[str]):
        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        # 拼接 prefix + text + suffix
        for idx, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][idx] = self.prefix_tokens + ids + self.suffix_tokens
        # pad 到 batch
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def _compute_logits(self, inputs) -> List[float]:
        logits = self.model(**inputs).logits[:, -1, :]
        true_score = logits[:, self.token_true_id]
        false_score = logits[:, self.token_false_id]
        batch_scores = torch.stack([false_score, true_score], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank_sorted(
            self,
            query: str,
            docs: List[str],
            instruction: Optional[str] = None,
            batch_size: int = 8,
    ) -> List[Tuple[int, float]]:
        """
        对单 query 多 doc 的场景，返回 [(doc_idx, score), ...] 按得分降序
        """
        queries = [query] * len(docs)
        pairs = [self._format(instruction, q, d) for q, d in zip(queries, docs)]

        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i: i + batch_size]
            inputs = self._tokenize(batch_pairs)
            scores.extend(self._compute_logits(inputs))

        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)


# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    # 1. 初始化模型（CPU 也能跑，GPU 更快）
    # model_name_or_path: 模型名称或本地路径。
    # -若为模型名称：如果本地缓存（默认路径~ /.cache/modelscope/）无该模型时，会自动从modelscope.cn下载模型权重、配置和分词器文件。
    #  如果本地缓存已存在该模型，代码会直接加载缓存中的文件，无需重新下载。
    # -若为本地模型的存储路径：则需要手动下载模型文件到路径地址（config.json、model.safetensors(或pytorch_model.bin)、tokenizer.json
    #  tokenizer_config.json、special_tokens_map.json、vocab.txt）.
    ranker = QwenReranker(
        # 由于本地硬件资源有限，替换为Qwen3-Reranker-0.6B模型.
        # model_name_or_path="Qwen/Qwen3-Reranker-4B",
        model_name_or_path="Qwen/Qwen3-Reranker-0.6B",
    )

    # 2. 测试查询+文档
    query = "人工智能在医疗领域的应用"
    docs = [
        "人工智能驱动的农业无人机可实现作物病虫害监测，精准率达85%以上",  # 无关（完全跨领域）
        "AI辅助诊断系统可快速分析CT、MRI等医学影像，使肺癌、乳腺癌等疾病的早期检出率提升30%以上",  # 高相关
        "深度学习算法在金融风控中的应用案例，有效降低信贷违约率15%",  # 无关（金融领域）
        "卷积神经网络在皮肤癌图像分类任务上达到91%的准确率，为基层医疗机构提供了低成本诊断方案",  # 中高相关（技术细分，间接相关）
        "智能模拟在生物医药研发中的应用，主要用于药物分子结构模拟，不属于临床医疗应用范畴",  # 干扰项（语义相近但领域不符）
        "人工智能技术在医疗领域的核心应用包括AI辅助诊断、个性化治疗方案生成、医疗影像分析等",  # 高相关
        "机器学习算法在医保风控系统中的应用，可识别虚假就医报销行为，属于AI在医疗管理的边缘场景",  # 低相关（跨领域AI，弱关联）
        "医疗大数据平台通过机器学习算法整合患者电子病历，为医院管理决策提供数据支撑",  # 中相关（泛医疗AI，关联性减弱）
    ]

    # 3. 自动排序
    print("\n【按得分降序】")
    sorted_res = ranker.rerank_sorted(query, docs)
    for rank, (idx, score) in enumerate(sorted_res, 1):
        print(f"{rank}. {score:.4f}\t{docs[idx]}")
