import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime


# 数据库初始化
def init_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT, amount REAL,
                 category TEXT, date TEXT)''')
    conn.commit()
    conn.close()


# 数据库操作函数
def get_all_records():
    conn = sqlite3.connect('data.db')
    df = pd.read_sql("SELECT * FROM records", conn)
    conn.close()
    return df


def update_record(record_id, name, amount, category):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("UPDATE records SET name=?, amount=?, category=? WHERE id=?",
              (name, amount, category, record_id))
    conn.commit()
    conn.close()


def delete_record(record_id):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("DELETE FROM records WHERE id=?", (record_id,))
    conn.commit()
    conn.close()


# 主界面
st.title("增强型数据管理系统")
init_db()

# 数据展示与操作区域
df = get_all_records()
if not df.empty:
    st.header("数据浏览")

    # 添加选择列
    df['选择'] = False
    edited_df = st.data_editor(
        df,
        column_config={
            "选择": st.column_config.CheckboxColumn(required=True),
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "date": st.column_config.DateColumn("日期", disabled=True)
        },
        hide_index=True,
        num_rows="fixed"
    )

    # 获取选中行
    selected_rows = edited_df[edited_df['选择']]
    if st.button("确认删除选中记录"):
        for _, row in selected_rows.iterrows():
            delete_record(row['id'])
        st.rerun()

    tab1, tab2 = st.tabs(["修改记录", "新增记录"])

    with tab1:
        with st.form("edit_form"):
            if not selected_rows.empty:
                record_id = selected_rows.iloc[0]['id']
                name = st.text_input("名称", value=selected_rows.iloc[0]['name'])
                amount = st.number_input("金额", value=selected_rows.iloc[0]['amount'])
                category = st.selectbox("分类", ["餐饮", "交通", "娱乐", "购物"],
                                        index=["餐饮", "交通", "娱乐", "购物"].index(selected_rows.iloc[0]['category']))

                if st.form_submit_button("更新记录"):
                    update_record(record_id, name, amount, category)
                    st.rerun()

    with tab2:
        # 新增记录区域
        with st.form("add_form"):
            name = st.text_input("名称")
            amount = st.number_input("金额", min_value=0.0)
            category = st.selectbox("分类", ["餐饮", "交通", "娱乐", "购物"])

            if st.form_submit_button("提交"):
                conn = sqlite3.connect('data.db')
                c = conn.cursor()
                c.execute("INSERT INTO records (name, amount, category, date) VALUES (?,?,?,?)",
                          (name, amount, category, datetime.now().strftime('%Y-%m-%d')))
                conn.commit()
                conn.close()
                st.rerun()
