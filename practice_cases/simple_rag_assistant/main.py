import os
import sys

import streamlit as st
from dotenv import load_dotenv

# 添加模块路径，由于导入的llm模块位于当前文件main.py的上上级目录。否则会报找不到module异常
# module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# 添加模块路径到sys.path中
# if module_path not in sys.path:
#     sys.path.append(module_path)

# 将当前脚本所在目录加入Python搜索路径（确保能找到services目录下的RAGService）
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from services.rag_service_stream import RAGService

# 加载环境变量
load_dotenv()

# 配置页面标题、图标、布局
st.set_page_config(
    page_title="RAG知识问答助手",
    page_icon=":robot:",
    layout="wide"
)


# 初始化Streamlit会话状态（跨刷新保存数据，避免页面刷新后数据丢失）
def initialize_app():
    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []

    # 用于重置文件上传框状态的会话变量
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0

    # 初始化RAG配置参数
    if "retrieve_k" not in st.session_state:
        st.session_state.retrieve_k = 6  # 默认检索文档数量
    if "enable_reranker" not in st.session_state:
        st.session_state.enable_reranker = False  # 默认开启重排
    if "rerank_top_n" not in st.session_state:
        st.session_state.rerank_top_n = 5  # 重排后保留文档数量

    # 初始化RAG核心服务（封装了文档处理、向量存储、流式问答的核心逻辑）
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RAGService(
            retrieve_k=st.session_state.retrieve_k,
            enable_reranker=st.session_state.enable_reranker,
            rerank_top_n=st.session_state.rerank_top_n,
            # 重排模型配置：用于对检索结果进行语义重排序.默认使用BAAI的bge-reranker-v2-m3（中文效果性价比较高）。
            # 这里加载本地路径模型（需手动下载模型文件到指定路径）
            # a.模型文件获取地址：https://modelscope.cn/models/BAAI/bge-reranker-v2-m3
            # b.需下载文件：config.json、model.safetensors、special_tokens_map.json、tokenizer.json、tokenizer_config.json
            model_name_or_path="../../data/models_reranker_data/BAAI/bge-reranker-v2-m3"  # 指定重排模型本地存储路径
        )


initialize_app()

# 定义侧边栏区域
with st.sidebar:
    st.subheader("RAG知识问答助手")

    # RAG检索配置区域
    # 1. 检索数量控制
    retrieve_k = st.slider(
        "初始检索文档数量 (retrieve_k)",
        min_value=1, max_value=10, value=st.session_state.retrieve_k, step=1,
        help="从向量库中初始检索的文档数量，数量越多覆盖范围越广，但可能引入噪音"
    )

    # 2. 重排功能开关
    enable_reranker = st.toggle(
        "开启检索结果重排",
        value=st.session_state.enable_reranker,
        help="开启后会对检索到的文档进行语义重排序，提升回答质量，但会增加响应时间"
    )

    # 3. 重排后保留数量（仅在开启重排时可配置）
    rerank_top_n = st.slider(
        "重排后保留文档数量 (rerank_top_n)",
        min_value=1, max_value=8, value=st.session_state.rerank_top_n, step=1,
        help="重排后最终保留的文档数量，需小于等于初始检索数量",
        disabled=not enable_reranker  # 关闭重排时禁用该参数
    )

    # 限制rerank_top_n不超过retrieve_k
    if rerank_top_n > retrieve_k:
        rerank_top_n = retrieve_k
        st.warning(f"重排保留数量自动调整为 {retrieve_k}（不超过初始检索数量）")

    # 4. 应用配置按钮
    if st.button("应用配置", use_container_width=True, type="primary"):
        # 更新会话状态
        st.session_state.enable_reranker = enable_reranker
        st.session_state.retrieve_k = retrieve_k
        st.session_state.rerank_top_n = rerank_top_n

        # 更新RAGService的配置
        st.session_state.rag_service.enable_reranker = enable_reranker
        st.session_state.rag_service.retrieve_k = retrieve_k
        st.session_state.rag_service.rerank_top_n = rerank_top_n

        st.success("配置已更新生效！")

    # 文档管理区域
    st.divider()  # 添加分隔线
    # 1. 多文件上传（支持PDF/DOCX/TXT/MD，与RAGService支持的格式一致）
    uploaded_files = st.file_uploader(
        "上传文档 (PDF/DOCX/txt/md)",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.upload_key}"  # 动态生成key
    )
    # 处理上传的文件：调用RAGService的process_document方法，完成“解析→分块→向量化→入库”
    if uploaded_files:
        with st.spinner("正在处理文档..."):
            for file in uploaded_files:
                st.session_state.rag_service.process_document(file)
            st.success(f"已成功处理 {len(uploaded_files)} 个文档")
            # 处理完成后重置上传框：通过改变key值实现
            st.session_state.upload_key += 1

    # 2. 清空知识库（删除向量库数据+清空聊天历史，重置整个问答环境）
    if st.button("清空知识库", type="secondary", use_container_width=True):
        with st.spinner("正在清空知识库..."):
            # 清空向量存储
            st.session_state.rag_service.clear_database()
            # 清空聊天历史
            st.session_state.history = []
            st.success("知识库已成功清空")

# 主界面 - 聊天区域
st.header("从0开始：用 Streamlit + LangChain搭建一个简单基于RAG问答聊天助手")

# 1. 展示聊天历史（遍历session_state.history，按角色显示消息）
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. 处理用户输入
user_input = st.chat_input("请问有什么可以帮助您？")
if user_input:
    # 步骤1：将用户消息添加到会话历史
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 步骤2：调用RAG服务生成流式回答，并显示
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # RAG回答，非流式：大模型完整输出后才展示出来
            # full_answer = rag_service.get_answer(user_input)
            # st.markdown(full_answer)

            # RAG回答，流式输出
            full_answer = ""  # 用于存储完整的回复内容
            # 调用RAGService的get_answer_stream（流式方法），用st.write_stream实现边生成边显示
            for chunk in st.write_stream(st.session_state.rag_service.get_answer_stream(user_input)):
                full_answer += chunk

            # 步骤3：将完整的助手回答添加到会话历史，供下次刷新时展示
            st.session_state.history.append({"role": "assistant", "content": full_answer})
            st.rerun()
