import argparse
from src.file_parser import parse_function_file, parse_calling_function_file, write_output_file
from llm_sdk.llm_sdk import Small_LLM_Model
from src.generator import generate_function_call
import json
from src.models import OutputFile, Function, FunctionCallingTest


def get_files_info() -> tuple[argparse.Namespace, list[Function], list[FunctionCallingTest]]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")

    args = parser.parse_args()
    functions = parse_function_file(args.functions_definition)
    tests = parse_calling_function_file(args.input)
    return (args, functions, tests)

def main() -> None:
    args, functions, tests = get_files_info()
    model = Small_LLM_Model()
    path = model.get_path_to_vocab_file()
    with open(path, "r") as f:
        vocab = json.load(f)
    
    outputs: list[OutputFile] = []
    for test in tests:
        output = generate_function_call(model, vocab, functions, test)
        outputs.append(output)

    write_output_file(args.output, outputs)


if __name__ == "__main__":
    main()