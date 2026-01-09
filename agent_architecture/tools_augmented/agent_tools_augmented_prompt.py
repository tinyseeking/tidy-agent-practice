from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from datetime import datetime
import logging

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载环境变量,读取.env文件配置信息
load_dotenv()

"""
模型调用配置说明：
1. 默认接入阿里云百炼「通义千问」系列；可无缝切换到任何 OpenAI SDK格式的模型服务。（如GPT系列、智谱清言、deepseek等）
2. 使用千问模型，需配置的环境变量：
   - QWEN_API_KEY: 模型API请求密钥，这里使用阿里云百炼API密钥（获取地址：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key）
   - QWEN_BASE_URL: 模型API服务地址（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
3. 若用其它模型服务或自行部署的模型，按需替换api_key、base_url值即可。
"""
api_key = os.getenv("QWEN_API_KEY")
if not api_key:
    raise ValueError("缺少 QWEN_API_KEY 环境变量（从阿里云控制台获取）")
base_url = os.getenv("QWEN_BASE_URL")
if not base_url:
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 构建模型调用客户端，支持所有兼容openai SDK的模型服务
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)


# 定义工具函数
def get_weather(location: str, date: str = None) -> dict:
    """获取指定城市、日期的天气信息"""
    # 实际应用中这里会调用真实的天气API
    # 为演示方便，返回模拟数据
    mock_data = {
        "北京": {
            "2025-05-26": {"condition": "晴", "temperature": "28℃", "humidity": "45%"},
            "2025-05-27": {"condition": "多云", "temperature": "26℃", "humidity": "50%"},
        },
        "上海": {
            "2025-05-26": {"condition": "小雨", "temperature": "24℃", "humidity": "70%"},
            "2025-05-27": {"condition": "阴", "temperature": "23℃", "humidity": "75%"},
        }
    }

    # 如果未提供日期，默认为今天
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    # 获取天气数据，如果城市或日期不存在，返回默认信息
    city_data = mock_data.get(location, {})
    weather_data = city_data.get(date, {"condition": "未知", "temperature": "未知", "humidity": "未知"})

    return {
        "location": location,
        "date": date,
        "condition": weather_data["condition"],
        "temperature": weather_data["temperature"],
        "humidity": weather_data["humidity"]
    }


def calculate(expression: str) -> float:
    """执行数学计算"""
    try:
        # 为安全起见，限制可执行的表达式
        allowed_ops = ['+', '-', '*', '/', '**', '(', ')', '.']
        if any(char not in allowed_ops + [str(i) for i in range(10)] for char in expression):
            raise ValueError("表达式包含不安全字符")

        # 使用eval计算表达式（实际应用中建议使用更安全的计算库）
        return eval(expression)
    except Exception as e:
        print(f"计算错误: {e}")
        return None


# 工具列表和对应的实现函数
TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate
}

# 工具描述（用于提示词）
TOOL_DESCRIPTIONS = """
    你可以使用以下工具解决问题：

    [工具1] 获取天气信息
    函数名：get_weather
    参数：
    - location：城市名称（字符串）
    - date：日期（格式：YYYY-MM-DD，默认为今天）
    返回：指定城市、日期的天气状况
    
    [工具2] 执行数学计算
    函数名：calculate
    参数：
    - expression：数学表达式（字符串）
    返回：表达式的计算结果
    
    如果你需要使用工具，请按以下JSON格式回复：
    {
      "name": "工具名",
      "parameters": {
        "参数名1": "值1",
        "参数名2": "值2"
      }
    }
"""


def parse_function_call(response_text: str) -> dict:
    """解析模型返回的文本，提取函数调用JSON"""
    try:
        # 简单的JSON提取逻辑，实际应用中可能需要更健壮的解析
        if "{" in response_text and "}" in response_text:
            start_idx = response_text.index("{")
            end_idx = response_text.rindex("}") + 1
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        return None
    except Exception as e:
        print(f"解析函数调用失败: {e}")
        return None


def call_llm(messages: list) -> str:
    """调用千问API获取回复"""

    logging.info("request messages: %s", messages)

    # 调用OpenAI API
    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        temperature=0  # 降低随机性，使输出更确定性
    )

    logging.info("response messages: %s", completion.choices[0].message.content)
    return completion.choices[0].message.content


def run_conversation(user_message: str) -> str:
    """运行完整的对话流程：用户提问 -> 模型生成工具调用 -> 执行工具 -> 模型生成最终回答"""
    # 构建包含工具描述的提示词
    system_prompt = f"""你是一个智能助手，可以使用工具解决用户的问题。
{TOOL_DESCRIPTIONS}

如果你需要调用工具，请按指定的JSON格式回复。
如果不需要调用工具，请直接回答用户问题。"""

    # 第一轮：获取用户问题并发送给模型
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # 调用千问API
    assistant_message = call_llm(messages)

    # 检查是否包含函数调用
    function_call = parse_function_call(assistant_message)

    if function_call:
        # 执行函数调用
        tool_name = function_call["name"]
        parameters = function_call.get("parameters", {})

        if tool_name in TOOLS:
            # 执行工具函数
            tool_function = TOOLS[tool_name]
            try:
                tool_result = tool_function(**parameters)

                # 将工具执行结果添加到对话中
                messages.append({"role": "assistant", "content": json.dumps(function_call)})
                messages.append({"role": "function", "name": tool_name, "content": json.dumps(tool_result)})

                # 再次调用模型，获取最终回答
                second_response = call_llm(messages)

                return second_response
            except Exception as e:
                return f"执行工具时出错: {e}"
        else:
            return f"未知工具: {tool_name}"
    else:
        # 如果模型没有调用工具，直接返回回答
        return assistant_message


# 示例使用
if __name__ == "__main__":
    # 示例1：天气查询
    user_question1 = "北京明天（2025-05-27）天气如何？"
    print(f"用户: {user_question1}")
    answer1 = run_conversation(user_question1)
    print(f"助手: {answer1}")

    # 示例2：数学计算
    user_question2 = "5*5*1000*333是多少？"
    print(f"\n用户: {user_question2}")
    answer2 = run_conversation(user_question2)
    print(f"助手: {answer2}")
