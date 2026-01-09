import os
import asyncio
import sys
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from openai import OpenAI

# 添加模块搜索路径，由于导入的llm模块位于当前文件的上上级目录。否则会报找不到module异常
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# 添加模块路径到sys.path中
if module_path not in sys.path:
    sys.path.append(module_path)


from llm.call_llm_struct import get_schema_json

# 加载环境变量,读取.env文件配置信息
load_dotenv()

# 初始化OpenAI客户端（适配千问/OpenAI/DeepSeek/智谱等兼容OpenAI接口规范的大模型）
# 替换说明：更换模型仅需调整 ①API密钥(api_key) ②服务地址(base_url) ③调用时指定的模型名称
client = OpenAI(
    # 千问模型API密钥（必填）：从环境变量读取，避免硬编码泄露
    # 官方文档：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    api_key=os.environ.get("QWEN_API_KEY"),

    # 千问API服务地址（兼容OpenAI格式）：从环境变量读取，适配不同部署环境
    # 默认值参考：https://dashscope.aliyuncs.com/compatible-mode/v1
    base_url=os.environ.get("QWEN_BASE_URL"),
)



# 定义响应模型
class VotingConfidenceResponse(BaseModel):
    confidence: float = Field(description="信心度分数，0表示没有问题，1表示有问题")


# 定义检查函数
async def sql_injection_check(query: str) -> Dict[str, Any]:
    """检查代码是否存在SQL注入漏洞"""
    # 构建包含JSON Schema的系统提示
    system_prompt = (
        "根据用户的输入信息检查是否存在SQL注入漏洞? 并严格按照以下JSON Schema返回JSON对象：\n"
        f"{get_schema_json(VotingConfidenceResponse)}\n\n"
        "注意：\n"
        "- 只返回符合Schema的JSON对象，不添加额外内容\n"
        "- 所有必需字段必须包含且类型正确\n"
        "- 日期应提取为字符串格式"
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    # 解析JSON响应
    import json
    try:
        parsed_response = json.loads(response.choices[0].message.content)
        # 验证响应格式
        validated_response = VotingConfidenceResponse(**parsed_response)
        return validated_response.model_dump()
    except json.JSONDecodeError:
        print(f"无法解析JSON响应: {response.choices[0].message.content}")
        return {"confidence": 0}
    except Exception as e:
        print(f"验证响应时出错: {e}")
        return {"confidence": 0}


async def exposed_secrets_check(query: str) -> Dict[str, Any]:
    """检查代码是否暴露任何密钥"""
    system_prompt = (
        "根据用户的输入信息检查是否会泄露敏感信息? 并严格按照以下JSON Schema返回JSON对象：\n"
        f"{get_schema_json(VotingConfidenceResponse)}\n\n"
        "注意：\n"
        "- 只返回符合Schema的JSON对象，不添加额外内容\n"
        "- 所有必需字段必须包含且类型正确\n"
        "- 日期应提取为字符串格式"
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    # 解析JSON响应
    import json
    try:
        parsed_response = json.loads(response.choices[0].message.content)
        validated_response = VotingConfidenceResponse(**parsed_response)
        return validated_response.model_dump()
    except json.JSONDecodeError:
        print(f"无法解析JSON响应: {response.choices[0].message.content}")
        return {"confidence": 0}
    except Exception as e:
        print(f"验证响应时出错: {e}")
        return {"confidence": 0}


async def proper_error_handling_check(query: str) -> Dict[str, Any]:
    """检查代码是否有适当的错误处理"""
    system_prompt = (
        "根据用户的输入信息检查函数是否具备恰当的错误处理机制? 并严格按照以下JSON Schema返回JSON对象：\n"
        f"{get_schema_json(VotingConfidenceResponse)}\n\n"
        "注意：\n"
        "- 只返回符合Schema的JSON对象，不添加额外内容\n"
        "- 所有必需字段必须包含且类型正确\n"
        "- 日期应提取为字符串格式"
    )
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    # 解析JSON响应
    import json
    try:
        parsed_response = json.loads(response.choices[0].message.content)
        validated_response = VotingConfidenceResponse(**parsed_response)
        return validated_response.model_dump()
    except json.JSONDecodeError:
        print(f"无法解析JSON响应: {response.choices[0].message.content}")
        return {"confidence": 0}
    except Exception as e:
        print(f"验证响应时出错: {e}")
        return {"confidence": 0}


async def parallel_query(query: str) -> Dict[str, Dict[str, Any]]:
    """并行执行多个安全检查"""
    # 并行执行所有检查
    sql_injection_task = sql_injection_check(query)
    exposed_secrets_task = exposed_secrets_check(query)
    proper_error_handling_task = proper_error_handling_check(query)

    # 等待所有任务完成
    sql_injection_result, exposed_secrets_result, proper_error_handling_result = await asyncio.gather(
        sql_injection_task, exposed_secrets_task, proper_error_handling_task
    )

    return {
        "sqlInjection": sql_injection_result,
        "exposedSecrets": exposed_secrets_result,
        "properErrorHandling": proper_error_handling_result
    }


def aggregator(responses: Dict[str, Dict[str, Any]]) -> None:
    """汇总并输出结果"""
    print("\n--- 代码安全分析结果 ---")
    for key, value in responses.items():
        confidence = value.get("confidence", 0)
        print(f"{key}: {confidence:.2f}")
        if confidence > 0.7:
            print(f"  ⚠️ 警告: 此代码可能存在{key}问题")
        elif confidence > 0.3:
            print(f"  🔍 注意: 此代码可能存在{key}问题")
        else:
            print(f"  ✅ 良好: 此代码没有明显的{key}问题")
    print("------------------------\n")


async def main() -> None:
    """主函数"""
    # 定义用户代码示例
    user_code_query = """
    def authenticate(username, password):
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        result = db.execute(query)
        return result is not None
    """
    print(f"分析代码：{user_code_query}")
    response = await parallel_query(user_code_query)
    aggregator(response)

    # 使用参数化查询，避免SQL注入
    user_code_query2 = """
    query = "SELECT * FROM users WHERE username = %s AND password = %s"
    result = db.execute(query, (username, password))
    return result is not None
    """
    print(f"分析代码：{user_code_query2}")
    response2 = await parallel_query(user_code_query2)
    aggregator(response2)

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
