# sse_client.py
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    # 连接到SSE服务器
    async with sse_client(url="http://localhost:8000/sse") as streams:
        async with ClientSession(*streams) as session:
            # 初始化会话
            await session.initialize()

            # 列出可用工具
            tools_response = await session.list_tools()
            print("Available tools:")
            for tool in tools_response.tools:
                print(f" - {tool.name}: {tool.description}")

            # 列出可用资源
            resources_response = await session.list_resources()
            print("\nAvailable resources:")
            for resource in resources_response.resources:
                print(f" - {resource.uri}: {resource.description}")

            # 调用天气工具
            print("\nCalling get_weather tool for London...")
            weather_response = await session.call_tool("get_weather", {"city": "London"})
            print(weather_response.content[0].text)

            # 读取资源
            print("\nReading weather://cities resource...")
            cities_response = await session.read_resource("weather://cities")
            print(cities_response[0].content)

            # 读取带参数的资源
            print("\nReading weather forecast for Tokyo...")
            forecast_response = await session.read_resource("weather://forecast/Tokyo")
            print(forecast_response[0].content)


if __name__ == "__main__":
    asyncio.run(main())
