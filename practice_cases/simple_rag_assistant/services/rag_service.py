import os
import tempfile
import shutil
from typing import List, Dict, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_chroma import Chroma

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from models.langchain_embedding import initialize_embedding_model
from models.langchain_llm import langchain_qwen_llm


class RAGService:
    """
    RAG（检索增强生成）服务类，实现文档解析、向量化存储及基于检索的知识进行问答，辅助 LLM 生成更准确、有依据的回答。
    核心流程：文档上传→解析分块→向量化存储→检索相关片段→LLM生成答案。
    """

    def __init__(self, persist_directory: str = "chroma_db"):
        """
        初始化RAG服务，加载嵌入模型、LLM模型及向量数据库。

        Args:
            persist_directory: 向量数据库持久化存储路径，默认值为"chroma_db"
        """
        # 向量数据库持久化目录
        self.persist_directory = persist_directory
        # 初始化嵌入模型（用于将文本转换为向量）
        self.embeddings = initialize_embedding_model("qwen")
        # 加载已存在的向量数据库（若存在）
        self.vectordb = self._load_vector_db()
        # 初始化大语言模型（用于生成答案）
        self.llm = langchain_qwen_llm()

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
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
            except Exception as e:
                raise RuntimeError(f"向量数据库加载失败（路径：{self.persist_directory}）：{str(e)}")
        return None

    def process_document(self, file: Any) -> Dict[str, bool | str]:
        """
        处理用户上传的文档（解析、分块、向量化、存储到向量数据库）。
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
        # -------------------------- （1）文件有效性校验与临时文件创建 --------------------------
        #  验证文件对象有效性
        if not file or not hasattr(file, 'name') or not hasattr(file, 'getvalue'):
            return {"success": False, "message": "无效的文件对象"}

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

            # -------------------------- （2）文档加载（按格式适配） --------------------------
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

            # -------------------------- （3）文本分块（解决长文本问题） --------------------------
            # 初始化文本分块器（解决长文本超出模型上下文窗口的问题）
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # 每个片段的字符数（根据模型上下文调整）
                chunk_overlap=200,  # 片段间重叠字符数（保持上下文连贯性）
                separators=["\n\n", "\n", "。", " ", ""]  # 优先按中文标点分割，提升分块合理性
            )
            # 将文档分割为片段（每个片段作为独立单元存入向量库）
            splits = text_splitter.split_documents(documents)

            # -------------------------- （4）向量存储 --------------------------
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

        except Exception as e:  # 捕获所有异常，返回具体错误信息
            return {"success": False, "message": f"文档处理失败（{file_name}）：{str(e)}"}
        finally:
            # -------------------------- （5）临时文件清理 --------------------------
            # 确保临时文件被清理（无论处理成功/失败）
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except Exception as e:
                    print(f"警告：临时文件清理失败（路径：{tmp_file_path}）：{str(e)}")

    def get_answer(self, question: str, chat_history: List[Dict[str, str]]) -> Optional[str]:
        """
        基于RAG技术生成问题答案：检索相关文档片段→结合对话历史 → 拼接提示词 → 调用LLM生成答案。

        Args:
            question: 用户当前的问题（字符串类型，非空）
            chat_history: 历史对话列表，格式为：
                [{"role": "user", "content": "用户问题"}, {"role": "assistant", "content": "助手回答"}, ...]

        Returns:
            生成的答案字符串；若发生错误，返回错误提示；若未上传文档，返回引导提示
        """
        # 1. 检查向量数据库是否初始化（是否已上传文档）
        if not self.vectordb:
            return "请先上传并处理文档，才能进行问答哦~"
        if not question or not isinstance(question, str) or question.strip() == "":
            return "请输入有效的问题内容~"

        # 2. 初始化对话记忆（存储历史对话，供LLM理解上下文）
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        # 加载历史对话到记忆中（转换为langchain支持的Message格式）
        try:
            for msg in chat_history:
                role = msg.get("role")
                content = msg.get("content", "").strip()
                if not role or not content:
                    continue
                if role == "user":
                    memory.chat_memory.add_message(HumanMessage(content=content))
                elif role == "assistant":
                    memory.chat_memory.add_message(AIMessage(content=content))
        except Exception as e:
            print(f"警告：历史对话加载失败，将忽略历史记录：{str(e)}")

        # 3. 创建向量数据库检索器，根据问题检索相关文档片段
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        # relevant_docs = retriever.get_relevant_documents(question)  # 直接调用检索方法
        relevant_docs = retriever.invoke(question)
        # 提取片段内容，格式化为字符串（便于拼接提示）
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 4. 拼接完整提示词
        # 从记忆中获取历史对话字符串（格式：Human: xxx\nAI: xxx）
        chat_history_str = memory.load_memory_variables({})["chat_history"]
        history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history_str])

        # 系统提示：定义LLM的角色和回答规则
        system_prompt = """
        你是基于文档的问答助手，仅使用以下提供的文档片段（Context）回答问题。
        如果文档中没有相关信息，直接说“根据提供的文档，无法回答该问题”，不要编造内容。
        回答需简洁、准确，结合历史对话（History）理解上下文，每一次回答要重新审视当前提供的内容，不要只是简单重复历史回答。
        
        Context:
        {context_text}
        
        History:
        {history_text}
        
        Current Question: {question}
        
        Answer:
        """

        # 填充变量，生成最终提示词
        final_prompt = system_prompt.format(
            context_text=context_text,
            history_text=history_text,
            question=question
        )
        print(f"组合提示词(final_prompt)：{final_prompt}")

        # 5. 调用LLM生成答案
        try:
            # # 直接调用LLM的生成方法
            # response = self.llm.generate([final_prompt])
            # # 提取生成结果（处理多输出情况，取第一个结果）
            # answer = response.generations[0][0].text.strip()
            # 使用invoke调用单个prompt（直接传入字符串，无需列表）
            response = self.llm.invoke(final_prompt)
            # 提取消息内容（content字段）
            answer = response.content.strip()
            # 将本次问答添加到记忆（供下次对话使用）
            memory.save_context({"question": question}, {"answer": answer})
            return answer
        except Exception as e:
            print(f"错误：答案生成失败：{str(e)}")
            return "抱歉，处理问题时发生错误，请稍后再试~"

    def clear_database(self) -> bool:
        """清空向量数据库（逻辑完全不变）"""
        try:
            if self.vectordb:
                self.vectordb.delete_collection()
                self.vectordb = None

            # if os.path.exists(self.persist_directory):
            #     shutil.rmtree(self.persist_directory)

            return True
        except Exception as e:
            print(f"错误：数据库清空失败：{str(e)}")
            return False
