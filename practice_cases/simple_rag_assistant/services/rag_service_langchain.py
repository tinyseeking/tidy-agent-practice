import os
import tempfile
import shutil
from typing import List, Dict, Optional, Any, Generator

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_chroma import Chroma

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

from models.langchain_embedding import initialize_embedding_model
from models.langchain_llm import langchain_qwen_llm


class RAGService:
    """
    RAG（检索增强生成）服务类，用于处理文档解析、向量存储及基于检索的问答功能。
    核心流程：文档上传→解析分块→向量存储→检索相关片段→LLM生成答案。
    """

    def __init__(self, persist_directory: str = "chroma_db"):
        """
        初始化RAG服务，加载嵌入模型、LLM模型及已存在的向量数据库。

        Args:
            persist_directory: 向量数据库持久化存储路径，默认值为"chroma_db"
        """
        # 向量数据库持久化目录
        self.persist_directory = persist_directory
        # 初始化嵌入模型（用于将文本转换为向量）
        self.embeddings = self._initialize_embedding_model("qwen")
        # 初始化大语言模型（用于生成答案）
        self.llm = self._initialize_llm()
        # 加载已存在的向量数据库（若存在）
        self.vectordb = self._load_vector_db()

    def _initialize_embedding_model(self, model_name: str) -> Any:
        """
        私有方法：初始化文本嵌入模型（封装实现，避免与主逻辑耦合）。
        嵌入模型用于将文本片段转换为向量，以便后续向量数据库存储和检索。

        Args:
            model_name: 嵌入模型名称（如"qwen"）

        Returns:
            初始化后的嵌入模型实例

        Raises:
            RuntimeError: 模型初始化失败时抛出异常（含具体错误信息）
        """
        try:
            # 实际使用时需替换为真实的嵌入模型初始化逻辑
            # 示例：from langchain.embeddings import HuggingFaceEmbeddings
            # return HuggingFaceEmbeddings(model_name=model_name)
            return initialize_embedding_model(model_name)  # 假设原初始化函数存在
        except Exception as e:
            raise RuntimeError(f"嵌入模型初始化失败（模型名：{model_name}）：{str(e)}")

    def _initialize_llm(self) -> Any:
        """
        私有方法：初始化大语言模型（LLM），用于基于检索到的文本生成答案。

        Returns:
            初始化后的LLM实例

        Raises:
            RuntimeError: LLM初始化失败时抛出异常（含具体错误信息）
        """
        try:
            # 假设initialize_qwen_llm()是项目中初始化Qwen模型的函数
            return langchain_qwen_llm()
        except Exception as e:
            raise RuntimeError(f"大语言模型初始化失败：{str(e)}")

    def _load_vector_db(self) -> Optional[Chroma]:
        """
        私有方法：加载已持久化的向量数据库（若目录存在且非空）。
        向量数据库用于存储文档片段的向量表示，支持高效的相似性检索。

        Returns:
            加载成功的Chroma向量数据库实例；若不存在或加载失败，返回None

        Raises:
            RuntimeError: 数据库加载过程中发生错误时抛出异常
        """
        # 检查持久化目录是否存在且非空
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            try:
                return Chroma(
                    embedding_function=self.embeddings,  # 指定嵌入模型（与存储时保持一致）
                    persist_directory=self.persist_directory  # 存储路径
                )
            except Exception as e:
                raise RuntimeError(f"向量数据库加载失败（路径：{self.persist_directory}）：{str(e)}")
        return None

    def process_document(self, file: Any) -> Dict[str, bool | str]:
        """
        处理用户上传的文档（解析、分块、存储到向量数据库）。
        支持的格式：PDF、DOCX、TXT、MD（可通过扩展loader支持更多格式）。

        Args:
            file: 上传的文件对象，需支持以下方法：
                - name: 属性，返回文件名（用于判断格式）
                - getvalue(): 方法，返回文件二进制内容（用于写入临时文件）

        Returns:
            处理结果字典，包含：
                - success: bool，处理是否成功
                - message: str，处理结果描述（成功时含片段数量，失败时含错误信息）
        """
        # 验证文件对象有效性
        if not file or not hasattr(file, 'name') or not hasattr(file, 'getvalue'):
            return {"success": False, "message": "无效的文件对象（需包含name属性和getvalue()方法）"}

        # 提取并标准化文件后缀（转为小写，便于格式判断）
        file_name = file.name
        file_suffix = file_name.split('.')[-1].lower() if '.' in file_name else ''
        tmp_file_path = None  # 临时文件路径（用于后续清理）

        try:
            # 创建临时文件存储上传的文件内容（避免直接操作内存中的二进制数据）
            with tempfile.NamedTemporaryFile(
                    delete=False,  # 关闭自动删除，确保加载器能读取
                    suffix=f".{file_suffix}",  # 保留文件后缀，避免加载器解析错误
                    mode='wb'  # 二进制写入模式
            ) as tmp_file:
                tmp_file.write(file.getvalue())  # 写入文件内容
                tmp_file_path = tmp_file.name  # 记录临时文件路径

            # 根据文件后缀选择对应的文档加载器
            if file_suffix == 'pdf':
                loader = PyPDFLoader(tmp_file_path)  # PDF加载器
            elif file_suffix == 'docx':
                loader = Docx2txtLoader(tmp_file_path)  # DOCX加载器
            elif file_suffix in ['txt', 'md']:
                loader = TextLoader(tmp_file_path, encoding='utf-8')  # 文本文件加载器（支持UTF-8编码）
            else:
                return {
                    "success": False,
                    "message": f"不支持的文件类型：{file_suffix}，当前支持：pdf/docx/txt/md"
                }

            # 加载文档内容（返回Document对象列表，每个对象含page_content和metadata）
            documents = loader.load()
            if not documents:  # 处理空文档情况
                return {"success": False, "message": "文档加载失败：内容为空或无法解析"}

            # 初始化文本分块器（解决长文本超出模型上下文窗口的问题）
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # 每个片段的字符数（根据模型上下文调整）
                chunk_overlap=200,  # 片段间重叠字符数（保持上下文连贯性）
                separators=["\n\n", "\n", "。", " ", ""]  # 优先按中文标点分割，提升分块合理性
            )
            # 将文档分割为片段（每个片段作为独立单元存入向量库）
            splits = text_splitter.split_documents(documents)

            # 将片段添加到向量数据库
            if self.vectordb:
                # 若数据库已存在，直接添加新片段
                self.vectordb.add_documents(splits)
            else:
                # 若数据库不存在，创建新库并添加片段
                self.vectordb = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,  # 使用初始化的嵌入模型
                    persist_directory=self.persist_directory  # 指定存储路径
                )

            return {
                "success": True,
                "message": f"文档处理成功！共添加 {len(splits)} 个文本片段（文件：{file_name}）"
            }

        except Exception as e:
            # 捕获所有异常，返回具体错误信息
            return {"success": False, "message": f"文档处理失败（{file_name}）：{str(e)}"}
        finally:
            # 确保临时文件被清理（无论处理成功/失败）
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except Exception as e:
                    # 仅打印警告，不影响主流程结果
                    print(f"警告：临时文件清理失败（路径：{tmp_file_path}）：{str(e)}")

    def get_answer(self, question: str, chat_history: List[Dict[str, str]]) -> Optional[str]:
        """
        基于RAG技术生成问题答案：检索相关文档片段→结合对话历史→LLM生成回答。

        Args:
            question: 用户当前的问题（字符串类型，非空）
            chat_history: 历史对话列表，格式为：
                [{"role": "user", "content": "用户问题"}, {"role": "assistant", "content": "助手回答"}, ...]

        Returns:
            生成的答案字符串；若发生错误，返回错误提示；若未上传文档，返回引导提示
        """

        # 检查问题有效性
        if not question or not isinstance(question, str) or question.strip() == "":
            return "请输入有效的问题内容~"

        # 初始化对话记忆（存储历史对话，供LLM理解上下文）
        # memory = ConversationBufferMemory(
        #     memory_key="chat_history",  # 与链中使用的键保持一致
        #     return_messages=True,  # 返回Message对象而非字符串，便于区分角色
        #     output_key="answer"  # 明确链的输出键，避免与其他键冲突
        # )

        # 初始化内存，设置窗口大小 k=50（只保留最近100轮对话）
        # ConversationBufferWindowMemory 是 ConversationBufferMemory 的扩展版本，专门用于解决长对话场景下的
        # 上下文管理问题。它通过只保留最近的 N 轮对话（滑动窗口机制），在维持对话连贯性的同时，避免历史记录过长导致的 Token 超限问题。
        memory = ConversationBufferWindowMemory(
            k=50,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # 加载历史对话到记忆中（转换为langchain支持的Message格式）
        try:
            for msg in chat_history:
                role = msg.get("role")  # 角色："user"或"assistant"
                content = msg.get("content", "").strip()  # 消息内容（去空）
                if not role or not content:
                    continue  # 跳过无效消息
                # 添加用户消息（HumanMessage）或助手消息（AIMessage）
                if role == "user":
                    memory.chat_memory.add_message(HumanMessage(content=content))
                elif role == "assistant":
                    memory.chat_memory.add_message(AIMessage(content=content))
        except Exception as e:
            # 历史对话加载失败不中断主流程，仅打印日志
            print(f"警告：历史对话加载失败，将忽略历史记录：{str(e)}")

        # 创建向量数据库检索器（用于根据问题检索相关文档片段）
        retriever = self.vectordb.as_retriever(
            search_kwargs={"k": 5}  # 检索最相关的5个片段（可根据效果调整）
        )

        # 构建对话式检索链（整合LLM、检索器、对话记忆）
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,  # 大语言模型
            retriever=retriever,  # 检索器
            memory=memory,  # 对话记忆
            return_source_documents=False,  # 暂不返回源文档（如需引用可设为True）
            verbose=False  # 生产环境关闭详细日志
        )

        # 调用链生成答案
        try:
            result = qa_chain.invoke({"question": question})  # 传入当前问题
            return result.get("answer", "抱歉，未能生成有效答案")
        except Exception as e:
            # 捕获生成过程中的异常，返回友好提示
            print(f"错误：答案生成失败：{str(e)}")
            return "抱歉，处理问题时发生错误，请稍后再试~"

    def clear_database(self) -> bool:
        """
        清空向量数据库（删除所有存储的文档片段及持久化文件）。
        谨慎使用：此操作不可逆，适用于重置数据库场景。

        Returns:
            bool: 清空操作是否成功
        """
        try:
            # 清除内存中的向量数据库
            if self.vectordb:
                self.vectordb.delete_collection()  # 删除集合
                self.vectordb = None  # 重置为None

            # 清除磁盘上的持久化文件
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)  # 递归删除目录及内容

            return True
        except Exception as e:
            print(f"错误：数据库清空失败：{str(e)}")
            return False
