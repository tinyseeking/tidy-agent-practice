import streamlit as st

st.title('streamlit 组件')

"# 1级标题"
"## 2级标题"
"### 3级标题"
"#### 4级标题"
"##### 5级标题"
"###### 6级标题"

"""
```python
print('代码片段')
```
"""

st.divider()

"### text_input"
name = st.text_input('请输入你的名字：')
if name:
    st.write(f'你好，{name}')

pwd = st.text_input('密码是多少？', type='password')

age = st.number_input('年龄：', value=20, min_value=0, max_value=200, step=1)
st.write(f'你输入的年龄是{age}岁')

"### text_area"
paragraph = st.text_area("多行内容：")

"### checkbox"
checked = st.checkbox("同意以上条款")
if checked:
    st.write("同意")
else:
    st.write("不同意")


"### radio"
res = st.radio(
    "给我的文章点个赞吧",
    ["点赞", "点多几篇"]
)

"### 下拉框 selectbox"
article = st.selectbox(
    "你喜欢哪篇文章？",
    [
        "《『Python爬虫』极简入门》",
        "《『SD』零代码AI绘画：光影字》",
        "《『SD』Stable Diffusion WebUI 安装插件（以汉化为例）》",
        "《NumPy入个门吧》"
    ]
)

st.write(f"你喜欢{article}")

"### 多选下拉框 multiselect"
article_list = st.multiselect("你喜欢哪篇文章？",
                              [
                                  "《『Python爬虫』极简入门》",
                                  "《『SD』零代码AI绘画：光影字》",
                                  "《『SD』Stable Diffusion WebUI 安装插件（以汉化为例）》",
                                  "《NumPy入个门吧》"
                              ]
                              )

for article in article_list:
    st.write(f"你喜欢{article}")

"### 滑块 slider"
height = st.slider("粉丝量", value=170, min_value=100, max_value=230, step=1)

st.write(f"你的粉丝量是{height}")

"### 按钮 button"
submitted = st.button("关注")

if submitted:
    st.write(f"用户点击关注啦啦啦啦啦")

"### 文件上传 file_uploader"
uploaded_file = st.file_uploader("上传文件", type=["csv", "json"])
if uploaded_file:
    st.write(f"你上传的文件是{uploaded_file.name}")

"### 多列布局 columns"
col1, col2, col3 = st.columns(3)

with col1:
    st.write('第1列')

with col2:
    st.write('第2列')

with col3:
    st.write('第3列')

"### 选项卡 tabs"
# 选项卡
tab1, tab2, tab3 = st.tabs(['点赞', '关注', '收藏'])

with tab1:
    st.write('快点赞吧')
with tab2:
    st.write('关注一下啦')
with tab3:
    st.write('收藏就是学会了')

"### 折叠展开组件 expander"
with st.expander('更多信息'):
    st.title('阳光')
    st.write('沙滩')
    st.write('仙人掌')
