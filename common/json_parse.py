import json


def is_json(myjson) -> bool:
    try:
        json.JSONDecoder().decode(myjson)
        return True
    except json.JSONDecodeError:
        return False
