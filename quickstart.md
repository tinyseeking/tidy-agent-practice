## 快速安装

### 一、设置虚拟环境

#### 1. 配置Conda 虚拟环境（安装 Conda，若未安装）

- 下载地址：https://docs.conda.io/en/latest/miniconda.html（conda 轻量化，推荐）

- 安装流程：

  - Windows：双击安装包，勾选 "Add Miniconda3 to PATH"（方便命令行调用）
  - Mac/Linux：执行安装脚本，按提示完成（默认会添加环境变量）

- 验证安装：打开终端 / 命令提示符，输入`conda --version`，显示版本号则成功

  ```
  D:\dev > conda --version
  conda 23.1.0
  ```

#### 2. 创建并激活 Conda 虚拟环境

1. 打开终端 / 命令提示符，执行以下命令创建虚拟环境（Python 版本指定 3.11）：

   ```bash
   # 创建虚拟环境(名称tidy-agent-practice-env可随意，也可使用其它已存在的虚拟环境)，指定Python 3.11
   conda create -n tidy-agent-practice-env python=3.11
   ```

   过程中会提示安装依赖，输入`y`确认。

2. 激活虚拟环境：

   - Windows（命令提示符）：

     ```cmd
     conda activate tidy-agent-practice-env 
     # 若执行不了，尝试：conda.bat activate tidy-agent-practice-env
     ```

   - Mac/Linux：

     ```bash
     conda activate tidy-agent-practice-env
     ```

   激活成功后，终端前缀会显示`(tidy-agent-practice-env)`

   如：

   ```
   (tidy-agent-practice-env) PS D:\dev\guwei\tidy-ai-lab\tidy-agent-practice> 
   ```
   
   

### 二、下载项目并安装依赖

#### **1. 下载项目**

```bash
git clone https://github.com/tidy/tidy-agent-practice.git
# 进入下载项目的根目录
cd tidy-agent-practice
```

#### 2. 安装依赖

这里介绍两种主要方式安装依赖：

##### **方法 1：**在 Conda 环境中用 `pip` 安装（最通用，兼容性最好）

1. 激活 Conda 环境

```bash
conda activate tidy-agent-practice-env
```

2. 用 `pip` 安装 `requirements.txt`

这是最传统的方式，利用 Conda 环境中内置的 `pip` 直接安装 `requirements.txt`，无需额外工具。

```bash
# 基础安装
pip install -r requirements.txt

# 国内用户可指定 PyPI 镜像加速，清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
##其它替代镜像源列表
# 阿里云镜像加速
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
# 中国科学技术大学镜像加速
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

**注意**：

- 确保 `pip` 属于当前 Conda 环境（执行 `which pip`（Linux/Mac）或 `where pip`（Windows），路径应指向 Conda 环境的 `bin/pip` 或 `Scripts/pip.exe`）。
- 若 Conda 环境中无 `pip`，可先执行 `conda install pip` 安装。

##### 方法 2：在 Conda 环境中用 `uv` 安装（推荐，速度最快）

`uv` 是新一代 Python 包管理工具，解析和安装 `requirements.txt` 的速度远快于 `pip`，且能完美兼容 Conda 环境。

**步骤 1：激活 Conda 环境**

```bash
conda activate tidy-agent-practice-env
```

**步骤 2：确保环境中安装了 `uv`**

如果未安装，可通过以下命令快速安装：

```bash
# 单二进制文件安装（推荐，不污染依赖）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或用 pip 安装
pip install uv
```

**步骤 3：用 `uv` 安装 `requirements.txt`**

```bash
# 极速安装依赖（自动识别 Conda 环境的 Python）
uv pip install -r requirements.txt

# 若需强制升级包，添加 --upgrade
uv pip install --upgrade -r requirements.txt
```

**优势**：解析速度极快、支持并行下载、能严格锁定依赖版本，是处理 `requirements.txt` 的最优选择。

### 三、环境变量设置

项目使用 `.env` 文件管理环境变量，主配置文件需放在项目根目录，部分子项目也可在其目录下单独创建 `.env` 文件。

操作方法：

1. 复制根目录的 `.env.template` 模板文件；
2. 重命名为 `.env`；
3. 按需修改文件内的配置参数。

修改配置说明如下：

```env
####################################
# 模型访问key及接口地址
####################################
# 千问模型接口访问key，默认;若使用其它模型，需配置相应的API_KEY和BASE_URL
# 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
QWEN_API_KEY="你的千问API Key"
# 千问模型接口访问地址
QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 智谱清言模型
ZHIPU_API_KEY="你的智谱API Key"
ZHIPU_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"

# 设置 deepseek 接口访问key,需要替换为你自己的
DEEPSEEK_API_KEY="你的DeepSeek API Key"
# 设置 deepseek 接口访问地址
DEEPSEEK_BASE_URL="https://api.deepseek.com"

# 设置 OPENAI 接口访问key,需要替换为你自己的
OPENAI_API_KEY="你的OpenAI API Key"
# 设置 OPENAI 接口访问地址
OPENAI_BASE_URL="https://api.openai.com/v1"

####################################
# 第三方服务接口
####################################
# tavily 搜索API KEY
# 访问 https://tavily.com 注册并获取API Key,用户每月有1000次免费调用额度.
TAVILY_API_KEY="你的Tavily API Key"

# 和风天气API配置,和风天气API文档地址 https://dev.qweather.com/docs/start/
# 请替换为你的KEY
HEFENG_API_KEY="a2add535d10***"
# 替换为你自己的和风 API Host。获取地址：https://console.qweather.com/setting?lang=zh
# API Host是开发者独立的API地址，用于取代传统的公共API地址
HEFENG_BASE_URL="https://mh7**.re.qweatherapi.com"
```

#### 各 API 密钥获取步骤

1. **千问模型（QWEN_API_KEY）**

   - 注册并登录阿里云账号

     1. 访问[阿里云官网](https://www.aliyun.com/)或直接进入[阿里云百炼平台](https://bailian.console.aliyun.com/#/home)，点击页面相关登录 / 注册入口。
     2. 可选择手机号注册、支付宝扫码等方式完成注册，已有阿里云账号的直接登录即可，若提示实名认证需按流程完成，否则无法开通后续服务。

   - 开通阿里云百炼服务

     登录后，如果页面顶部显示提示"立即开通"，表示您需开通阿里云百炼模型服务，以获得[新人免费额度](https://help.aliyun.com/zh/model-studio/new-free-quota)。开通阿里云百炼不会产生费用，仅模型调用（超出免费额度后）、模型部署、模型调优会产生相应计费。如果页面顶部未显示开通提示，则表示您已经开通。

   - 创建并获取 API - Key

     1. 开通服务后进入百炼控制台，找到并进入**密钥管理**页面（模型服务->密钥管理）。
     2. 切换到 “API - Key” 页签，点击**创建 API - KEY**按钮。
     3. 在弹出的弹窗中，选择 API Key 的归属账号（主账号或子账号）和归属业务空间（一般选默认业务空间即可），填写简要描述后点击确定。
     4. 创建成功后，点击该 API Key 旁的复制图标，即可获取完整密钥，建议立即妥善保存，避免泄露。

   - 官方文档：https://bailian.console.aliyun.com/?tab=api#/api

2. **智谱清言模型（ZHIPU_API_KEY）**

   - 访问智谱 AI 开放平台：https://open.bigmodel.cn/
   - 注册并登录账号
   - 进入 "控制台"，点击左侧 "API 密钥"
   - 点击 "创建 API 密钥"，输入密钥名称
   - 生成后即可获取 API Key（格式为`xxxxxxxx.xxxxxx`）
   - 注意：免费用户有调用额度限制，超出需付费

3. **DeepSeek 模型（DEEPSEEK_API_KEY）**

   - 访问 DeepSeek 开放平台：https://www.deepseek.com/
   - 注册并登录账号
   - 进入 "开发者中心" 或 "API 控制台"
   - 在 "API keys" 页面点击 "创建 API key"
   - 生成后复制 API Key 并保存
   - 注意查看免费额度及使用规范

4. **OpenAI 模型（OPENAI_API_KEY）**

   - 官方渠道（需科学上网）：
     - 访问 OpenAI 官网：https://platform.openai.com/
     - 注册并登录账号
     - 点击右上角头像，选择 "View API keys"
     - 点击 "Create new secret key"，输入名称
     - 生成后复制并保存密钥

5. **Tavily 搜索服务（TAVILY_API_KEY）**

   - 访问 Tavily 官网：https://tavily.com/
   - 注册并登录账号（支持邮箱或 Google 账号）
   - 登录后进入 "Dashboard"
   - 在 "API Keys" 页面可直接获取默认密钥，或点击 "Generate new key" 创建
   - 免费用户每月有 1000 次调用额度，超出需升级付费方案

6. **和风天气API**

   - **一、获取 API 密钥（KEY）**

     1. 注册并登录[和风开发者控制台](https://console.qweather.com/project)。
     2. 进入「项目管理」→「创建项目」，填写名称，提交后进入项目详情页。
     3. 在项目详情页中，点击右侧的“添加凭据”按钮。输入凭据名称、选择身份认证方式（API KEY）,点击“保存”按钮，即创建凭据。
     4. 复制 API KEY，妥善保管密钥。密钥泄露应删除凭据，再重新创建。

   - **二、获取专属 API Host（独立域名）**

     1. 登录控制台 → 进入「设置」，复制你的专属 API Host（格式如 `xxx.re.qweatherapi.com`）。

   - **三、构造调用地址（HTTPS 强制）**

     1. 完整 URL 格式：`https://你的API Host/v7/接口路径?key=你的KEY&location=城市ID/经纬度`。
     2. 示例（实况天气）：`https://xxx.re.qweatherapi.com/v7/weather/now?key=xxx&location=101010100`。
     3. 小程序 / APP 需将 API Host 加入网络请求白名单（如微信小程序域名白名单）。

   - **四、关键提醒**

     - 双重认证：请求需同时携带「专属 API Host + API Key」，缺一不可。

     - 免费版频限 10 次 / 分钟、日 3000 次，仅 HTTPS 请求有效。


## 运行说明

项目中的代码根据是否包含交互界面，分为**Streamlit UI 界面运行**和**命令行直接运行**两种方式：

- 带 Streamlit 交互界面的代码：使用 `streamlit run 文件相对路径` 命令启动；
- 无界面的代码：直接使用 `python 文件相对路径` 命令运行。

以下是运行示例：

#### **1. 记忆管理测试（Streamlit 界面）**

```bash
streamlit run agent_architecture/memory/chat_memory_app.py
```

**功能**：提供可视化聊天界面，可在侧边栏自由切换裁剪、总结、分层记忆等多种记忆策略，测试不同记忆模式下对话上下文的处理效果。

![image-20251218173529119](./tutorials/images/image-20251218173529119.png)

#### 2. ReAct 认知策略测试（命令行）

```bash
python cognitive_architecture/react/react_langgrath.py
```

**功能**：演示 ReAct 智能体的核心工作流程，模拟 “思考 - 行动 - 观察” 的循环过程，包含天气查询、数学计算等工具调用示例，支持流式输出智能体的思考与操作步骤。

#### 3.简单RAG 实战（simple_rag_assistant）

在项目根目录下，打开终端执行命令：

```
cd simple_rag_assistant
streamlit run main.py
```

![image-20260104144926176](./tutorials/images/image-20260104144926176.png)

系统将启动 Web 服务，默认地址为 [http://localhost:8501](http://localhost:8501/)

![image-20260104160926112](./tutorials/images/image-20260104160926112.png)
