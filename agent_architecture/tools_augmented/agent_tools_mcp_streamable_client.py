import asyncio
import json
import os
import sys
from typing import Any, Optional, Dict, List, Tuple

import mcp
from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

# 添加模块搜索路径，由于导入的llm及common模块位于当前文件main.py的上上级目录。否则会报找不到module异常
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# 添加模块路径到sys.path中
if module_path not in sys.path:
    sys.path.append(module_path)


from common import logger
from cfastmcp.mcp_to_openai_tools import fastmcp_to_openai_tools

"""
模型调用配置说明：
1. 默认接入阿里云百炼「通义千问」系列；可无缝切换到任何 OpenAI SDK格式的模型服务。（如GPT系列、智谱清言、deepseek等）
2. 使用千问模型，需配置的环境变量：
   - QWEN_API_KEY: 模型API请求密钥，这里使用阿里云百炼API密钥（获取地址：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key）
   - QWEN_BASE_URL: 模型API服务地址（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
3. 若用其它模型服务或自行部署的模型，按需替换api_key、base_url值即可。
"""

# 加载环境变量
load_dotenv()


# 环境变量加载与验证
def load_env_config() -> Tuple[str, str]:
    """加载并验证环境变量配置"""
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("缺少 QWEN_API_KEY 环境变量（从阿里云控制台获取）")
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return api_key, base_url


logger = logger.configure_logging(__name__)


class LLMClient:
    """LLM客户端，负责与大语言模型API通信"""

    def __init__(self, model_name: str, base_url: str, api_key: str) -> None:
        """
        初始化LLM客户端

        Args:
            model_name: 模型名称（如qwen-plus、gpt-3.5-turbo等）
            api_key: 模型API密钥
            base_url: 模型API请求地址
        """
        self.model_name: str = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_model_response(self, messages: list[dict[str, str]], tools: List[Dict[str, Any]]) -> ChatCompletionMessage:
        """
        发送消息给LLM并获取响应

        Args:
            messages: 对话消息列表
            tools: OpenAI格式的工具定义列表

        Returns:
            LLM响应消息
        """
        logger.info(f"请求模型消息：\n{messages}")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,  # 降低随机性，使输出更确定性,
            tools=tools if tools else None,
            tool_choice="auto",  # 让模型自动决定是否调用工具
            stream=False
        )
        return response.choices[0].message


class MCPClient:
    """MCP 客户端封装类，用于与 MCP 服务器交互"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 MCP 客户端

        Args:
            config: MCP 服务器配置，如果为 None 则使用默认配置
        """
        # 默认配置（指定 Streamable HTTP 传输协议）
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


class ChatSession:
    """聊天会话，处理用户输入和LLM响应，并与MCP工具交互"""

    def __init__(self, llm_client: LLMClient, mcp_client: MCPClient) -> None:
        self.llm_client: LLMClient = llm_client
        self.mcp_client: MCPClient = mcp_client
        # 工具定义（符合Function Calling规范）
        self.tools: List[Dict[str, Any]] = []

    async def _init_mcp_tools(self):
        """异步初始化MCP工具"""
        try:
            # 获取MCP Server注册的工具列表
            mcp_tools = await self.mcp_client.list_tools()
            # 将 MCP Server返回的工具列表转换为OpenAI 工具调用格式
            self.tools = fastmcp_to_openai_tools(mcp_tools)
            print(f"成功加载 {len(self.tools)} 个MCP工具")
        except Exception as e:
            print(f"初始化MCP工具失败: {str(e)}")
            self.tools = []

    @staticmethod
    def parse_tool_calls(message: ChatCompletionMessage) -> List[Dict[str, Any]]:
        """
        解析LLM响应中的工具调用（符合OpenAI Function Calling规范）

        Args:
            message: LLM响应消息

        Returns:
            工具调用列表，格式：[{"name": 工具名, "parameters": 参数字典}, ...]
        """
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    function = tool_call.function
                    params = json.loads(function.arguments) if function.arguments else {}
                    tool_calls.append({
                        "name": function.name,
                        "parameters": params,
                        "id": tool_call.id
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"解析工具调用参数失败: {str(e)}")
                    continue
        return tool_calls

    async def process_call_tool(self, tool_name: str, arguments: Any) -> str:
        """
        通过MCP Client执行工具调用

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具调用结果（字符串格式）
        """
        try:
            # 验证工具是否存在
            mcp_tools = await self.mcp_client.list_tools()
            tool_exists = any(tool.name == tool_name for tool in mcp_tools)
            if not tool_exists:
                error_msg = f"工具 {tool_name} 不存在"
                return json.dumps({"error": error_msg})

            logger.info(f"执行工具: {tool_name}, 参数: {arguments}")
            result = await self.mcp_client.call_tool(tool_name, arguments)
            logger.info(f"工具 {tool_name} 执行结果: {result}")

            # 记录来源信息
            if result and len(result.content) > 0:
                retrieve_text = result.content[0].model_dump().get("text")
                # # 解析为JSON对象
                # parsed_data = json.loads(retrieve_text)
                # logger.info(f"工具 {tool_name} 结果解析完成 | 解析后数据：{parsed_data}")
                return retrieve_text
            return ""  # 但返回空结果
        except Exception as e:
            error_msg = f"工具调用失败: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

    async def run_conversation(self, system_message) -> None:
        """
        启动聊天会话的主循环

        Args:
            system_message: 系统提示词
        """
        # 会话前，先初始化MCP工具
        await self._init_mcp_tools()
        print("欢迎使用天气助手！可查询天气、时间信息，输入'exit'可退出程序。")

        # 添加系统提示词到消息记录中
        messages = [{"role": "system", "content": system_message}]

        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入您的问题：")
                if not user_input:
                    print("助手: 请输入有效的问题或指令")
                    continue
                if user_input in ["quit", "exit", "退出"]:
                    print("助手: 再见！")
                    break

                # 添加用户输入到对话历史
                messages.append({"role": "user", "content": user_input})

                # 工具调用循环（控制最大的工具调用次数,防止无限调用）
                max_function_calls = 5
                function_call_count = 0
                while function_call_count < max_function_calls:
                    # 获取LLM响应
                    message = self.llm_client.get_model_response(messages, self.tools)
                    # 显示助手回答（非工具调用部分）
                    if message.content:
                        print(f"助手: {message.content}")

                    # 解析工具调用
                    tool_calls = self.parse_tool_calls(message)
                    has_tool_call = bool(tool_calls)
                    if has_tool_call:
                        function_call_count += 1
                        logger.info(f"第 {function_call_count} 次工具调用，共 {len(tool_calls)} 个工具需要调用")

                        # 记录助手的工具调用意图
                        messages.append({
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": message.tool_calls  # 保存工具调用信息
                        })

                        # 处理每个工具调用
                        for tool_call in tool_calls:
                            tool_name = tool_call["name"]
                            parameters = tool_call["parameters"]

                            # 执行工具调用
                            tool_result = await self.process_call_tool(tool_name, parameters)

                            # 添加工具调用结果到对话历史
                            messages.append({
                                "role": "tool",
                                "name": tool_name,
                                "tool_call_id": tool_call["id"],
                                "content": tool_result
                            })
                    else:
                        # 无工具调用，记录最终回答
                        messages.append({"role": "assistant", "content": message.content})
                        break

                if function_call_count >= max_function_calls:
                    print("助手: 已达到最大工具调用次数，本次对话结束")
                    messages.append({"role": "assistant", "content": "已达到最大工具调用次数，无法继续处理你的请求。"})

            except Exception as e:
                logger.error(f"会话处理异常: {str(e)}")
                print(f"助手: 处理你的请求时发生错误: {str(e)}")
                # 清空当前用户输入的影响，避免错误累积
                messages.pop()


async def main():
    """主函数"""
    try:
        # 初始化客户端
        api_key, base_url = load_env_config()
        llm_client = LLMClient(model_name="qwen-plus-latest", api_key=api_key, base_url=base_url)

        # mcp 连接配置
        mcp_config = {
            "mcpServers": {
                "customer_service": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable-http"
                }
            }
        }
        mcp_client = MCPClient(mcp_config)

        # 构建聊天会话
        chat_session = ChatSession(llm_client=llm_client, mcp_client=mcp_client)
        # 系统提示，指导LLM如何使用工具和返回响应
        system_message = "你是一个智能助手，能够使用提供的工具解决用户问题。"
        # 启动会话
        await chat_session.run_conversation(system_message=system_message)
    except Exception as e:
        logger.error(f"程序启动失败: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
