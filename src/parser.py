import json
from models import Function, FunctionCallingTest


def parse_function_file(filename: str) -> list[Function]:
    with open(filename, "r") as f:
        data = json.load(f)

    return [Function(**item) for item in data]


def parse_calling_function_file(filename: str) -> list[FunctionCallingTest]:
    with open(filename, "r") as f:
        data = json.load(f)

    return [FunctionCallingTest(**item) for item in data]