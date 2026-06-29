import json
import os
from src.models import Function, FunctionCallingTest, OutputFile
from pydantic import ValidationError


def parse_function_file(filename: str) -> list[Function]:
    """Parse a functions definition JSON file.

    Args:
        filename: Path to the JSON file.

    Returns:
        List of Function objects.
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        if not data:
            raise SystemExit(f"ERROR: Empty file: {filename}")

        return [Function(**item) for item in data]

    except json.JSONDecodeError:
        raise SystemExit(f"ERROR: Invalid JSON in file: {filename}")

    except FileNotFoundError:
        raise SystemExit(f"ERROR: File not found: {filename}")

    except ValidationError as e:
        raise SystemExit(f"ERROR: Invalid function definition: {e}")


def parse_calling_function_file(
    filename: str,
) -> list[FunctionCallingTest]:
    """Parse a function calling tests JSON file.

    Args:
        filename: Path to the JSON file.

    Returns:
        List of FunctionCallingTest objects.
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        if not data:
            raise SystemExit(f"ERROR: Empty file: {filename}")
        return [FunctionCallingTest(**item) for item in data]

    except json.JSONDecodeError:
        raise SystemExit(f"ERROR: Invalid JSON in file: {filename}")

    except FileNotFoundError:
        raise SystemExit(f"ERROR: File not found: {filename}")

    except ValidationError as e:
        raise SystemExit(f"ERROR: Invalid function definition: {e}")


def write_output_file(
    file_path: str,
    output_files: list[OutputFile],
) -> None:
    """Write output results to a JSON file.

    Args:
        file_path: Path to the output file.
        output_files: List of OutputFile objects to write.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = [item.model_dump() for item in output_files]
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: File not found: {file_path}")
