from typing import List, Dict, Any

from langchain_core.tools import StructuredTool, BaseTool
from mcp import ClientSession
from mcp.types import (
    Tool as MCPTool, CallToolResult, TextContent, ImageContent, EmbeddedResource,
)


async def load_mcp_langchain_tools(session: ClientSession) -> list[BaseTool]:
    """
    从 MCP 会话加载可用工具并转换为 LangChain 工具列表

    Args:
        session: MCP 客户端会话对象

    Returns:
        转换后的 LangChain 工具列表
    """
    # 获取 MCP 服务器提供的工具列表
    mcp_tools = await session.list_tools()
    print(mcp_tools)

    # 存储转换后的 LangChain 工具
    langchain_tools = [
        convert_mcp_tool_to_langchain_tool(session, tool)
        for tool in mcp_tools.tools
    ]

    return langchain_tools


def convert_mcp_tool_to_langchain_tool(
        session: ClientSession | None,
        tool: MCPTool,
) -> BaseTool:
    """Convert an MCP tool to a LangChain tool.

    NOTE: this tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session
        tool: MCP tool to convert
        connection: Optional connection config to use to create a new session
                    if a `session` is not provided

    Returns:
        a LangChain tool
    """

    async def call_tool(
            **arguments: dict[str, Any],
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        call_tool_result = await session.call_tool(tool.name, arguments)
        return call_tool_result.content

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=call_tool,
        response_format="content",
        metadata=tool.annotations.model_dump() if tool.annotations else None,
    )
