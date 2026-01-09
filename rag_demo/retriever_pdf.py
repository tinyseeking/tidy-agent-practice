import json
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import  PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from llm.langchain.langchain_embedding import langchain_embedding_model

tmp_file_path="./docs/智联未来科技有限公司.pdf"
# 加载网页
loader = PyPDFLoader(tmp_file_path)
docs = loader.load()

# 分割文本为块

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# 设置嵌入模型
embeddings = langchain_embedding_model("qwen")

# 设置持久化目录
persist_directory = r'./chroma_db'

# 创建向量数据库
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory
)
# print(vectordb._collection.count())

# 查询向量数据库
# question = "你的公司叫什么"
question = "公司的发展历程"
docs = vectordb.similarity_search(question, k=5)
# print(len(docs))
for doc in docs:
    print(doc.page_content)
    print(json.dumps(doc.metadata))

# 使用检索器查询相关文档
print("**********get_relevant_documents**********")
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
rel_docs = retriever.get_relevant_documents(question)
for rel_doc in rel_docs:
    print(rel_doc.page_content)
    print(json.dumps(rel_doc.metadata))


