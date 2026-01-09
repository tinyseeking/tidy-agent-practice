import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 创建STDIO服务器连接参数对象
# StdioServerParameters 是 FastMCP (Micro Controller Protocol) 框架中的类，
# 用于配置基于标准输入输出 (STDIO) 通信的服务器连接参数。
# 这个类主要用于定义如何启动和连接到一个通过标准输入输出进行通信的外部进程。
server_params = StdioServerParameters(
    command="python",  # 启动Python解释器
    args=["agent_tools_mcp_stdio_server.py"],  # 执行agent_tools_mcp_stdio_server.py脚本
    # 可选参数示例
    # cwd="/path/to/workdir",
    # env={"ENV_VAR": "value"},
    encoding="utf-8",
    # startup_timeout=30.0
)


async def main():
    async with stdio_client(server_params) as (read, write):
        # 创建客户端会话，用于与服务器进行交互
        # sampling_callback为None表示不使用采样回调功能
        async with ClientSession(
                read, write, sampling_callback=None
        ) as session:
            # 初始化会话，通常用于建立协议、认证等准备工作
            await session.initialize()

            # 获取服务器端可用的工具列表
            tool = await session.list_tools()
            print(f"可用工具列表:\n")
            for idx, tool in enumerate(tool.tools, 1):
                print(f"  {idx}. 工具名：{tool.name} | {tool.description}| {tool.inputSchema}")

            print('\n调用工具 calculate,计算"188*23-34"')
            # 调用名为"calculate"的工具，传入数学表达式作为参数
            result = await session.call_tool("calculate", {"expression": "188*23-34"})

            # 打印工具调用结果内容
            print(f"执行结果：{result.content[0].model_dump().get('text')}")


if __name__ == "__main__":
    asyncio.run(main())
