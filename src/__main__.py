import argparse
from src.file_parser import parse_function_file, parse_calling_function_file, write_output_file


parser = argparse.ArgumentParser()

parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
parser.add_argument("--input", default="data/input/function_calling_tests.json")
parser.add_argument("--output", default="data/output/function_calls.json")

args = parser.parse_args()
functions = parse_function_file(args.functions_definition)
tests = parse_calling_function_file(args.input)
write_output_file(args.output, [])