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

# 创建MCP服务器实例
mcp = FastMCP(name="mcp_assistant_streamable_Server")


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
    """
    # 1. 搜索查询城市，获取城市ID
    city_search = search_city_info(query=location, number=2)
    logger.debug(f"city_search result :{city_search}")
    if not city_search.get("error") and city_search["results"]:
        location_id = city_search["results"][0]["location_id"]
        logger.debug(f"找到 ID: {location_id}")

        # 2. 根据城市ID获取当前天气
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
                    "text_day": "白天天气状况",
                    "text_night": "夜间天气状况",
                    "wind_dir_day": "白天风向",
                    "wind_scale_day": "白天风力",
                    "precip": "降水量概率",
                    "humidity": "相对湿度"
                },
                ...
            ]
        }
    """
    # 1. 搜索查询城市，获取城市ID
    city_search = search_city_info(query=location, number=2)
    logger.debug(f"city_search result :{city_search}")
    if not city_search.get("error") and city_search["results"]:
        location_id = city_search["results"][0]["location_id"]
        logger.debug(f"找到 ID: {location_id}")

        # 2. 根据城市ID获取当前天气
        forecast_weather = get_forecast_weather(location_id, days)
        logger.debug(f"获取未来天气: {forecast_weather}")
        return forecast_weather
    return {"error": f"为查询到相关城市数据： location:{location}"}


if __name__ == "__main__":
    # 使用 Streamable HTTP传输方式启动服务器
    mcp.run(transport="streamable-http",  # ② 直接换关键字
            host="0.0.0.0",
            port=8000,
            path="/mcp")
