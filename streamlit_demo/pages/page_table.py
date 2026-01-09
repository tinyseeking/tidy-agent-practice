import streamlit as st
import pandas as pd

# 示例数据
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [24, 30, 18],
    'city': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)


# 添加操作列（删除按钮）
def delete_row(index, df=None):
    df = df.drop(index)
    st.session_state['df'] = df  # 更新 session state 中的 DataFrame
    st.rerun()  # 重新运行应用以反映更改


# 添加一个 session state 变量来存储 DataFrame，以便在会话中保持状态
if 'df' not in st.session_state:
    st.session_state['df'] = df.copy()  # 初始化 DataFrame 在 session state 中

# 显示表格和按钮
for index, row in st.session_state['df'].iterrows():
    st.divider()
    cols = st.columns(4)  # 创建三列用于显示数据和按钮
    cols[0].write(row['name'])
    cols[1].write(row['age'])
    cols[2].write(row['city'])
    with cols[3]:  # 在第三列中添加删除按钮
        if st.button("Delete", key=f"delete_{index}"):
            delete_row(index,df)  # 调用删除函数并传递行索引
