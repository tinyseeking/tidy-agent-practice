import mcp.types


def fastmcp_to_openai_functions(mcp_tools: list[mcp.types.Tool], additionalProperties: bool = False) -> list:
    """
    将MCP Server返回的工具列表转换为OpenAI函数调用格式

    参数：
        mcp_tools: MCP工具列表，每个元素需包含name, description, inputSchema字段
        additionalProperties: 决定是否允许对象包含除了已定义属性之外的其他属性,False:不允许额外属性,True:允许

    返回：
        符合OpenAI函数调用规范的JSON对象列表
    """

    function_definitions = []  # 用来存放转换后的工具列表

    for tool in mcp_tools:  # 遍历每一个MCP工具
        # 第一步：创建OpenAI工具的基本框架
        function_schema = {
            "name": tool.name,  # 工具名称，比如"search_weather"
            "description": tool.description,  # 工具描述，让AI知道这个工具是干什么的
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": additionalProperties
            }  # 参数部分，稍后会填充
        }

        # 第二步：获取MCP工具的参数定义
        input_schema = tool.inputSchema  # 这里包含了工具需要什么参数

        # 第三步：转换参数格式
        parameters = {
            "type": input_schema['type'],  # 参数类型，通常是"object"（对象）
            "properties": input_schema['properties'],  # 具体的参数定义
            "required": input_schema.get('required', []),  # 哪些参数是必须的
            "additionalProperties": False  # 不允许额外参数，这是安全考虑
        }

        # 第四步：特殊处理枚举类型参数
        # 如果参数有固定的可选值，我们要让AI更清楚地知道
        for prop in parameters["properties"].values():
            if "enum" in prop:  # 如果这个参数有枚举值（比如颜色只能是红、绿、蓝）
                # 把可选值写进描述里，让AI更容易理解
                prop["description"] = prop["description"] + f",可选值: {', '.join(prop['enum'])}"

        # 第五步：把处理好的参数放回工具定义中
        function_schema["parameters"] = parameters
        function_definitions.append(function_schema)  # 添加到结果列表

    return function_definitions  # 返回转换完成的工具列表


def fastmcp_to_openai_tools(mcp_tools: list[mcp.types.Tool], additionalProperties: bool = False) -> list:
    """
    将 MCP Server返回的工具列表转换为OpenAI 工具调用格式。
    (`tools` 是 `functions` 的升级版，提供更强大的功能和更灵活的结构，推荐使用该种方式。)

    参数：
        mcp_tools: MCP工具列表，每个元素需包含name, description, inputSchema字段
        additionalProperties: 决定是否允许对象包含除了已定义属性之外的其他属性,False:不允许额外属性,True:允许

    返回：
        符合OpenAI函数调用规范的JSON对象列表
    """

    openai_tools = []  # 用来存放转换后的工具列表

    for tool in mcp_tools:  # 遍历每一个MCP工具
        # 第一步：创建OpenAI工具的基本框架
        tool_schema = {
            "type": "function",  # 告诉OpenAI这是一个函数工具
            "function": {
                "name": tool.name,  # 工具名称，比如"search_weather"
                "description": tool.description,  # 工具描述，让AI知道这个工具是干什么的
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": additionalProperties
                }  # 参数部分，稍后会填充
            }
        }

        # 第二步：获取MCP工具的参数定义
        input_schema = tool.inputSchema  # 这里包含了工具需要什么参数

        # 第三步：转换参数格式
        parameters = {
            "type": input_schema['type'],  # 参数类型，通常是"object"（对象）
            "properties": input_schema['properties'],  # 具体的参数定义
            "required": input_schema.get('required', []),  # 哪些参数是必须的
            "additionalProperties": False  # 不允许额外参数，这是安全考虑
        }

        # 第四步：特殊处理枚举类型参数
        # 如果参数有固定的可选值，我们要让AI更清楚地知道
        for prop in parameters["properties"].values():
            if "enum" in prop:  # 如果这个参数有枚举值（比如颜色只能是红、绿、蓝）
                # 把可选值写进描述里，让AI更容易理解
                prop["description"] = f"可选值: {', '.join(prop['enum'])}"

        # 第五步：把处理好的参数放回工具定义中
        tool_schema["function"]["parameters"] = parameters
        openai_tools.append(tool_schema)  # 添加到结果列表

    # print("\nconverte to openai tools :", [openai_tools])
    return openai_tools  # 返回转换完成的工具列表
