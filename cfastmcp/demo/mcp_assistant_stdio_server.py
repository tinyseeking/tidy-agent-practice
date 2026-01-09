# calculator_stdio.py
import datetime
import json
from typing import Any

import requests
from fastmcp import FastMCP

# 创建MCP服务器实例
mcp = FastMCP("mcp_assistant_stdio_Server")


@mcp.tool()
def calculate(expression: str) -> Any | None:
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


@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个数字相加"""
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """从第一个数中减去第二个数"""
    return a - b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """将两个数相乘"""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """将第一个数除以第二个数"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


@mcp.tool(
    name="get_current_time",
    description="获取当前时间",
)
def get_current_time():
    """
       获取当前时间并进行格式化展示
       :return:
    """
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


@mcp.tool(
    name="get_location",
    description="获取当前地点",
)
def get_location():
    """
       获取当前地点
       :return:
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
    # 使用stdio传输方式启动服务器
    mcp.run(transport="stdio")
