import logging

import streamlit as st

import os
import sys

# # 添加模块路径，由于导入的call_llm及common模块位于当前文件main.py的上上级目录。否则会报找不到module异常
# module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# # 添加模块路径到sys.path中
# if module_path not in sys.path:
#     sys.path.append(module_path)

# 设置页面配置
st.set_page_config(
    page_title="streamlit Demo",
    page_icon=":robot_face:",
    layout="wide"
)

# 定义导航页面
page_component = st.Page("pages/page_home.py", title="组件")
page_table = st.Page("pages/page_table.py", title="静态表格")
page_crud = st.Page("pages/page_crud.py", title="记账管理(crud)")
page_curd2 = st.Page("pages/page_crud2.py", title="记账管理增强(crud)")

pg = st.navigation([page_component,page_table, page_crud, page_curd2])
pg.run()
