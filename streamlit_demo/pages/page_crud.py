import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime


# 初始化数据库
def init_db():
    conn = sqlite3.connect('demo_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT, amount REAL, 
                 category TEXT,
                  date DATETIME)''')
    conn.commit()
    conn.close()


# 增删改查操作
def add_record(name, amount, category):
    conn = sqlite3.connect('demo_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO records (name, amount, category, date) VALUES (?,?,?,?)",
              (name, amount, category, datetime.now().strftime('%Y-%m-%d')))
    conn.commit()
    conn.close()


def get_all_records():
    conn = sqlite3.connect('demo_data.db')
    df = pd.read_sql("SELECT * FROM records", conn)
    conn.close()
    return df


def update_record(record_id, new_data):
    conn = sqlite3.connect('demo_data.db')
    c = conn.cursor()
    c.execute("UPDATE records SET name=?, amount=?, category=? WHERE id=?",
              (new_data['name'], new_data['amount'], new_data['category'], record_id))
    conn.commit()
    conn.close()


def delete_record(record_id):
    conn = sqlite3.connect('demo_data.db')
    c = conn.cursor()
    c.execute("DELETE FROM records WHERE id=?", (record_id,))
    conn.commit()
    conn.close()


# 界面布局
st.title("账本管理")
init_db()

tab1, tab2 = st.tabs(["数据浏览", "数据操作"])

with tab1:
    df = get_all_records()
    df['选择'] = False
    edited_df = st.data_editor(
        df,
        column_config={
            "选择": st.column_config.CheckboxColumn(required=True),
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "name": st.column_config.Column("名称", disabled=True),
            "date": st.column_config.Column("日期", disabled=True),
            "category": st.column_config.Column("目录", disabled=True),
            "amount": st.column_config.NumberColumn("金额", format="￥%.2f", disabled=True),
        },
        hide_index=True,
        key="id",
        # num_rows="dynamic",
        num_rows="fixed"
    )

    if st.button("保存修改"):
        for idx, row in edited_df.iterrows():
            update_record(row['id'], row)
        st.success("数据更新成功！")
    if st.button("删除选中行", type="primary"):
        selected_rows = edited_df[edited_df['选择']]
        if not selected_rows.empty:
            for index, row in selected_rows.iterrows():
                delete_record(row['id'])
            st.rerun()
with tab2:
    with st.form("新增记录"):
        name = st.text_input("名称")
        amount = st.number_input("金额", min_value=0.0)
        category = st.selectbox("分类", ["餐饮", "交通", "娱乐", "购物"])

        if st.form_submit_button("添加记录"):
            add_record(name, amount, category)
            st.rerun()

