import json
from src.models import Function, FunctionCallingTest, OutputFile


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


def write_output_file(file_path: str, output_files: list[OutputFile]):
    try:
        data = [item.model_dump() for item in output_files]

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    except FileNotFoundError:
        raise SystemExit("ERROR: File Not Found.")