import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# StdioServerParameters 是 FastMCP (Micro Controller Protocol) 框架中的一个类，
# 用于配置基于标准输入输出 (STDIO) 通信的服务器连接参数。
# 这个类主要用于定义如何启动和连接到一个通过标准输入输出进行通信的外部进程。
server_params = StdioServerParameters(
    command="python",  # 启动Python解释器
    args=["mcp_assistant_stdio_server.py"],  # 执行server.py脚本
    # 可选参数示例
    # cwd="/path/to/workdir",
    # env={"ENV_VAR": "value"},
    # encoding="utf-8",
    # startup_timeout=30.0
)


async def main():
    async with stdio_client(server_params) as (read, write):
        # 创建一个客户端会话，用于与服务器进行交互
        # sampling_callback为None表示不使用采样回调功能
        async with ClientSession(
                read, write, sampling_callback=None
        ) as session:
            # 初始化会话，通常用于建立协议、认证等准备工作
            await session.initialize()

            # 获取服务器端可用的工具列表
            tools = await session.list_tools()
            print(f"Available tools: {tools}")

            print('\n正在调用工具...')

            # 调用名为"calculate"的工具，传入数学表达式作为参数
            result = await session.call_tool("calculate", {"expression": "188*23-34"})

            # 打印工具调用结果内容
            print(result.content)


asyncio.run(main())