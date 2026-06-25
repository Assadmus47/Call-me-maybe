import json
from models import Function, FunctionCallingTest


def parse_function_file(filename: str) -> list[Function]:
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        return [Function(**item) for item in data]

    except json.JSONDecodeError:
        raise SystemExit(f"ERROR: Invalid JSON in file: {filename}")
    except FileNotFoundError:
        raise SystemExit("ERROR: File Not Found.")


def parse_calling_function_file(filename: str) -> list[FunctionCallingTest]:
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        return [FunctionCallingTest(**item) for item in data]

    except json.JSONDecodeError:
        raise SystemExit(f"ERROR: Invalid JSON in file: {filename}")
    except FileNotFoundError:
        raise SystemExit("ERROR: File Not Found.")