import datetime
import json
from typing import Any

import requests
from fastmcp import FastMCP

# ==============================================
# 核心初始化：创建MCP服务器实例
# ==============================================
# 实例化FastMCP对象，参数为服务器名称（自定义标识，用于区分不同MCP服务器）
mcp = FastMCP("mcp_assistant_stdio_Server")


# ==============================================
# 工具函数1：数学计算功能
# ==============================================
@mcp.tool(
    name="calculate",
    description="执行数学表达式计算",
)
def calculate(expression: str) -> Any | None:
    """
    执行安全受限的数学表达式计算
    参数:
        expression: str - 需要计算的数学表达式字符串（如"1+2*3"）
    返回:
        Any | None - 计算结果（成功时返回数值，失败时返回None）
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


# ==============================================
# 工具函数2：获取当前时间
# ==============================================
@mcp.tool(
    name="get_current_time",  # 显式指定工具名称（若不指定则默认使用函数名）
    description="获取当前时间",  # 工具描述，用于告知调用方该函数的功能
)
def get_current_time():
    """
       获取当前时间并进行格式化展示
    """
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


# ==============================================
# 工具函数3：获取当前地理位置（基于IP）
# ==============================================
@mcp.tool(
    name="get_location",
    description="获取当前地理位置信息",
)
def get_location():
    """
    功能：通过IP地址获取当前地理位置信息
    返回值：JSON格式的字符串，包含国家/地区/城市信息（成功）或错误信息（失败）
    """
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()

        if data["status"] == "success":
            location_info = {
                "country": data.get("country", ""),
                "region": data.get("regionName", ""),
                "city": data.get("city", "")
            }
            return json.dumps(location_info, ensure_ascii=False)
        else:
            return json.dumps({"error": "无法获取地理位置"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    # 启动MCP服务器，使用stdio（标准输入输出）作为通信方式
    mcp.run(transport="stdio")
