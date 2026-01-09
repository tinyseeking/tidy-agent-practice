import os

import requests
import json
from typing import Dict, Any

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 和风天气API配置（请替换为你的KEY）
# 和风天气API地址 https://dev.qweather.com/docs
hefeng_api_key = os.getenv("HEFENG_API_KEY")
if not hefeng_api_key:
    raise ValueError("缺少HEFENG_API_KEY环境变量")
hefeng_base_url = os.getenv("HEFENG_BASE_URL")

if not hefeng_base_url:  # 默认百炼模型调用服务端口
    raise ValueError("缺少HEFENG_BASE_URL环境变量")


def get_current_weather(location_id: str) -> Dict[str, Any]:
    """
    获取指定城市的实时天气情况，包括实时温度、体感温度、风力风向、相对湿度、大气压强、降水量、能见度、露点温度、云量等

    Args:
        location_id: 城市Location ID（可通过search_city获取）或以英文逗号分隔的经度,纬度坐标，例如 location=101010100 或 location=116.41,39.92

    Returns:
        包含当前天气信息的字典，结构如下：
        {
            "location": {"name": "城市名称", "id": "城市ID"},
            "update_time": "更新时间",
            "temp": "当前温度",
            "feels_like": "体感温度",
            "text": "天气状况描述",
            "wind_dir": "风向",
            "wind_scale": "风力等级",
            "humidity": "相对湿度",
            "pressure": "气压",
            "visibility": "能见度",
            "precip": "降水量"
        }
    """
    url = f"{hefeng_base_url}/v7/weather/now"
    params = {
        "location": location_id
    }

    # 添加请求头
    headers = {
        "X-QW-Api-Key": hefeng_api_key
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # 检查HTTP状态码
        data = response.json()

        if data["code"] != "200":
            raise ValueError(f"API错误: {data['code']} - {data.get('message', '未知错误')}")

        now = data["now"]
        return {
            "location": {"name": data.get("locationName", location_id), "id": location_id},
            "update_time": data.get("updateTime", ""),
            "temp": now.get("temp", ""),
            "feels_like": now.get("feelsLike", ""),
            "text": now.get("text", ""),
            "wind_dir": now.get("windDir", ""),
            "wind_scale": now.get("windScale", ""),
            "humidity": now.get("humidity", ""),
            "pressure": now.get("pressure", ""),
            "visibility": now.get("vis", ""),
            "precip": now.get("precip", "")
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求错误: {str(e)}"}
    except (KeyError, json.JSONDecodeError) as e:
        return {"error": f"数据解析错误: {str(e)}"}
    except ValueError as e:
        return {"error": str(e)}


def get_forecast_weather(location_id: str, days: int = 7) -> Dict[str, Any]:
    """
    获取指定城市的未来几天天气情况

    Args:
        location_id: 城市Location ID
        days: 预报天数（支持最多30天预报，默认7天），可选值：3、7、10、15、30

    Returns:
        包含未来天气信息的字典，结构如下：
        {
            "location": {"name": "城市名称", "id": "城市ID"},
            "update_time": "更新时间",
            "daily": [
                {
                    "date": "日期",
                    "temp_min": "最低温度",
                    "temp_max": "最高温度",
                    "text_day": "白天天气状况",
                    "text_night": "夜间天气状况",
                    "wind_dir_day": "白天风向",
                    "wind_scale_day": "白天风力",
                    "precip": "降水量概率",
                    "humidity": "相对湿度"
                },
                ...
            ]
        }
    """
    if days not in range(1, 30):
        return {"error": "预报天数必须在1-30天之间"}

    url = f"{hefeng_base_url}/v7/weather/{days}d"
    params = {
        "location": location_id
    }

    # 添加请求头
    headers = {
        "X-QW-Api-Key": hefeng_api_key
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if data["code"] != "200":
            raise ValueError(f"API错误: {data['code']} - {data.get('message', '未知错误')}")

        daily_forecast = []
        for day in data.get("daily", []):
            daily_forecast.append({
                "date": day.get("fxDate", ""),
                "temp_min": day.get("tempMin", ""),
                "temp_max": day.get("tempMax", ""),
                "text_day": day.get("textDay", ""),
                "text_night": day.get("textNight", ""),
                "wind_dir_day": day.get("windDirDay", ""),
                "wind_scale_day": day.get("windScaleDay", ""),
                "precip": day.get("precip", ""),
                "humidity": day.get("humidity", "")
            })

        return {
            "location": {"name": data.get("locationName", location_id), "id": location_id},
            "update_time": data.get("updateTime", ""),
            "daily": daily_forecast
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求错误: {str(e)}"}
    except (KeyError, json.JSONDecodeError) as e:
        return {"error": f"数据解析错误: {str(e)}"}
    except ValueError as e:
        return {"error": str(e)}


def search_city_info(query: str, language: str = "zh", number: int = 2) -> Dict[str, Any]:
    """
    根据关键词获取城市基本信息，包括城市的Location ID，多语言名称、经纬度、时区、海拔、Rank值、所在行政区域等。

    Args:
        query: 搜索关键词（城市名称、拼音等）
        language: 返回语言（默认中文：zh-CN）
        number: 返回结果数量（默认2）

    Returns:
        包含城市搜索结果的字典，结构如下：
        {
            "search_query": "搜索关键词",
            "results": [
                {
                    "id": "城市Location ID",
                    "name": "城市名称",
                    "name_en": "英文名称",
                    "adm1": "上级行政区",
                    "adm2": "所属行政区",
                    "country": "所属国家",
                    "lat": "纬度",
                    "lon": "经度",
                    "tz": "时区",
                    "alt": "海拔",
                    "rank": "排名值"
                },
                ...
            ]
        }
    """
    url = f"{hefeng_base_url}/geo/v2/city/lookup"
    params = {
        "location": query,
        "lang": language,
        "number": number
    }

    # 添加请求头
    headers = {
        "X-QW-Api-Key": hefeng_api_key
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        if data["code"] != "200":
            raise ValueError(f"API错误: {data['code']} - {data.get('message', '未知错误')}")

        cities = []
        for city in data.get("location", []):
            cities.append({
                "location_id": city.get("id", ""),
                "name": city.get("name", ""),
                "name_en": city.get("name_en", ""),
                "adm1": city.get("adm1", ""),
                "adm2": city.get("adm2", ""),
                "country": city.get("country", ""),
                "lat": city.get("lat", ""),
                "lon": city.get("lon", ""),
                "tz": city.get("tz", ""),
                "alt": city.get("alt", ""),
                "rank": city.get("rank", "")
            })

        return {
            "search_query": query,
            "results": cities[:2] if len(cities) > 2 else cities
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求错误: {str(e)}"}
    except (KeyError, json.JSONDecodeError) as e:
        return {"error": f"数据解析错误: {str(e)}"}
    except ValueError as e:
        return {"error": str(e)}


# 示例用法
if __name__ == "__main__":
    # 1. 搜索城市
    city_search = search_city_info(query="上海",number=2)
    print(f"city_search result :{city_search}")
    if not city_search.get("error") and city_search["results"]:
        location_id = city_search["results"][0]["location_id"]
        print(f"找到 ID: {location_id}")

        # 2. 获取当前天气
        current_weather = get_current_weather(location_id)
        print(f"当前天气: {current_weather.get('text', '未知')}, {current_weather.get('temp', '未知')}°C")

        # 3. 获取未来天气
        forecast = get_forecast_weather(location_id, 7)
        print(f"未来天气预报:")
        for day in forecast.get("daily", []):
            print(f" {day}")
    else:
        print(f"城市搜索失败: {city_search.get('error', '未知错误')}")
