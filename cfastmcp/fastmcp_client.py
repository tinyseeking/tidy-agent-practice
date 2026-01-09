import asyncio
from typing import Any, Dict, List, Optional
from fastmcp import Client  # FastMCP 客户端
import mcp.types


class MCPClient:
    """MCP 客户端封装类，用于与 MCP 服务器交互"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 MCP 客户端

        Args:
            config: MCP 服务器配置，如果为 None 则使用默认配置
        """
        # 默认配置
        default_config = {
            "mcpServers": {
                "customer_service": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable-http"
                }
            }
        }
        # 合并配置（用户传入的配置优先级更高）
        self.config = config or default_config
        # 初始化 FastMCP 客户端
        self.client = Client(self.config)

    async def list_tools(self) -> List[mcp.types.Tool]:
        """
        异步获取 MCP Server 注册的工具列表

        Returns:
            工具列表
        """
        async with self.client:
            tools = await self.client.list_tools()
            return tools

    def list_tools_sync(self) -> List[mcp.types.Tool]:
        """
        同步获取 MCP Server 注册的工具列表

        Returns:
            工具列表
        """
        return asyncio.run(self.list_tools())

    async def call_tool(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        """
        异步调用服务端注册的工具

        Args:
            name: 工具名称
            args: 工具调用参数（可选）

        Returns:
            工具调用结果
        """
        async with self.client:
            if args:
                result = await self.client.call_tool(name, args)
            else:
                result = await self.client.call_tool(name)
            return result

    def call_tool_sync(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        """
        同步调用服务端注册的工具

        Args:
            name: 工具名称
            args: 工具调用参数（可选）

        Returns:
            工具调用结果
        """
        return asyncio.run(self.call_tool(name, args))

    def __del__(self):
        """析构函数，确保客户端资源释放"""
        try:
            # 可根据 FastMCP 客户端的实际情况添加资源释放逻辑
            pass
        except Exception:
            pass


# 使用示例
if __name__ == "__main__":
    # 1. 配置创建客户端
    custom_config = {
        "mcpServers": {
            "customer_service": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable-http"
            },
            # "local_service": {
            #     "command": "python",
            #     "args": ["./customer_service.py"]
            # }
        }
    }
    mcp_client = MCPClient(custom_config)

    # 2. 同步获取工具列表
    tools = mcp_client.list_tools_sync()
    print(f"注册的工具列表: {[tool.name for tool in tools]}")

    # 3. 同步调用工具（示例）
    # result = mcp_client.call_tool_sync("tool_name", {"key": "value"})
    # print(f"工具调用结果: {result}")

