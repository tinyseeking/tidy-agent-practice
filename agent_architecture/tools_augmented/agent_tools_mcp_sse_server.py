import os
import sys
from typing import Any, Dict

from fastmcp import FastMCP

# 添加模块搜索路径，由于导入的llm及common模块位于当前文件main.py的上上级目录。否则会报找不到module异常
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# 添加模块路径到sys.path中
if module_path not in sys.path:
    sys.path.append(module_path)

# 使用和风天气API获取天气信息，文档地址：https://dev.qweather.com/docs/start/
# 需要配置环境变量 HEFENG_API_KEY(和风API密钥)、HEFENG_BASE_URL(开发者独立的API调用地址)
from tools.weather.weather_hefeng import search_city_info, get_current_weather, get_forecast_weather

from common import logger

# 日志记录配置
logger = logger.configure_logging(logger_name=__name__)


# 初始化FastMCP实例，命名为"mcp_assistant_sse_Server"
mcp = FastMCP(name="mcp_assistant_sse_Server")


# 定义工具函数
@mcp.tool()
def calculate(expression: str) -> Any | None:
    """
    执行安全的数学表达式计算（仅支持白名单内的字符和运算）

    Args:
        expression (str): 待计算的数学表达式，例如 "1+2*3"

    Returns:
        Any | None: 计算结果（数字类型），若表达式非法或计算出错则返回 None
    """
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


@mcp.tool()
def fetch_current_weather(location: str) -> dict:
    """
    获取指定城市的当日天气信息

    Args:
        location (str): 城市名称，例如 "北京"、"上海"

    Returns:
        Dict[str, Any]: 天气信息字典，包含以下两种情况：
        - 成功：返回天气数据（如温度、湿度、天气状况等）
        - 失败：返回 {"error": 错误信息}

    核心逻辑：
        1. 先通过城市名称搜索获取城市ID（和风API必须通过ID查询天气）
        2. 再通过城市ID调用实时天气接口获取数据
    """
    # 步骤1：搜索城市信息，获取城市ID（number=2 表示返回前2个匹配结果，提高容错）
    city_search = search_city_info(query=location, number=2)
    logger.debug(f"city_search result :{city_search}")
    if not city_search.get("error") and city_search["results"]:
        # 取第一个匹配结果的城市ID（优先匹配最相关的城市）
        location_id = city_search["results"][0]["location_id"]
        logger.debug(f"找到 ID: {location_id}")

        # 步骤2：根据城市ID获取实时天气数据
        curr_weather = get_current_weather(location_id)
        logger.debug(f"获取当前天气: {curr_weather}")
        return curr_weather
    return {"error": f"为查询到相关城市数据： location:{location}"}


@mcp.tool()
def fetch_forcast_weather(location: str, days: int = 7) -> Dict[str, Any]:
    """
    获取指定城市的未来几天天气情况

    Args:
        location (str): 城市名称，例如 "北京"、"上海"
        days: 预报天数（支持最多30天预报，默认7天），可选值：3、7、10、15、30

    Returns:
        包含未来天气信息的字典，结构如下：
        {
            "location": {"name": "城市名称", "id": "城市ID"},
            "update_time": "更新时间",
            "daily": [
                {
                    "date": "日期",
                    "temp_min": "最低温度",
                    "temp_max": "最高温度",
                      ...
                },
                ...
            ]
        }
    """
    # 步骤1：搜索城市信息，获取城市ID
    city_search = search_city_info(query=location, number=2)
    logger.debug(f"city_search result :{city_search}")
    if not city_search.get("error") and city_search["results"]:
        location_id = city_search["results"][0]["location_id"]
        logger.debug(f"找到 ID: {location_id}")

        # 步骤2：根据城市ID和指定天数获取天气预报
        forecast_weather = get_forecast_weather(location_id, days)
        logger.debug(f"获取未来天气: {forecast_weather}")
        return forecast_weather
    return {"error": f"为查询到相关城市数据： location:{location}"}


# 程序入口：当直接运行该脚本时，启动MCP服务
if __name__ == "__main__":
    # 启动FastMCP服务，配置说明：
    # - transport='sse': 使用SSE（Server-Sent Events）传输协议，适用于实时单向推送
    # - host="0.0.0.0": 监听所有网络接口，允许外部访问（仅建议测试/内网环境）
    # - port=8001: 服务端口号，需确保端口未被占用
    # 启动后，工具函数可通过SSE协议被客户端调用
    mcp.run(transport='sse', host="0.0.0.0", port=8001)