from src.models import OutputFile, Function, FunctionCallingTest
from llm_sdk.llm_sdk import Small_LLM_Model
from src.file_parser import parse_function_file, parse_calling_function_file
from src.constrained_decoder import *
from src.prompt_builder import prompt_builder


def get_function_name(model, vocab, functions_names, ids_list) -> str:
    generated_so_far: str = ""
    while (generated_so_far not in functions_names):
        logits = model.get_logits_from_input_ids(ids_list)

        logits = find_valide_token(logits, vocab, functions_names, generated_so_far)

        max_id = logits.index(max(logits))
        generated_so_far += model.decode([max_id])
        ids_list.append(max_id)
    
    return generated_so_far


def generate_function_call(model: Small_LLM_Model, vocab: dict[str, float], functions: list[Function], test: FunctionCallingTest) -> OutputFile:
    prompt = prompt_builder(functions, test)

    ids = model.encode(prompt)

    ids_list = ids[0].tolist()
    generated_so_far = ""
    functions_names = [function.name for function in functions]

    generated_so_far = get_function_name(model, vocab, functions_names, ids_list)
    function = find_function_name(functions, generated_so_far)

    if function is None:
        raise SystemExit("ERROR: Function not found")

    fixed_ids = model.encode('", "parameters": {')[0].tolist()
    ids_list += fixed_ids
    params = {}

    for i, (param_name, param_info) in enumerate(function.parameters.items()):
        if i == 0:
            prefix = f'"{param_name}": '
        else:
            prefix = f', "{param_name}": '
        fixed_ids = model.encode(prefix)[0].tolist()
        ids_list += fixed_ids

        if param_info.type == "number":
            value, ids_list = generate_number_value(model, vocab, ids_list)
        elif param_info.type == "string":
            value, ids_list = generate_string_value(model, vocab, ids_list)
        elif param_info.type == "bool":
            value, ids_list = generate_boolean_value(model, vocab, ids_list)
        
        params[param_name] = value

    return OutputFile(prompt=test.prompt, name=generated_so_far, parameters=params)