from typing import Any
from src.models import OutputFile, Function, FunctionCallingTest
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constrained_decoder import (
    find_valide_token,
    find_function_name,
    generate_number_value,
    generate_string_value,
    generate_boolean_value,
)
from src.prompt_builder import prompt_builder


def get_function_name(
    model: Small_LLM_Model,
    vocab: dict[str, Any],
    functions_names: list[str],
    ids_list: list[int],
) -> str:
    """Generate the function name using constrained decoding.

    Args:
        model: The LLM model instance.
        vocab: The vocabulary dictionary.
        functions_names: List of valid function names.
        ids_list: Current list of token IDs.

    Returns:
        The generated function name.
    """
    generated_so_far: str = ""
    counter = 0
    while generated_so_far not in functions_names:
        logits = model.get_logits_from_input_ids(ids_list)
        logits = find_valide_token(
            logits, vocab, functions_names, generated_so_far
        )
        max_id = logits.index(max(logits))
        generated_so_far += model.decode([max_id])
        ids_list.append(max_id)
        counter += 1
        if counter == 50:
            raise SystemExit("ERROR: Could not generate function name")
    return generated_so_far


def generate_function_call(
    model: Small_LLM_Model,
    vocab: dict[str, Any],
    functions: list[Function],
    test: FunctionCallingTest,
) -> OutputFile:
    """Generate a function call from a natural language prompt.

    Args:
        model: The LLM model instance.
        vocab: The vocabulary dictionary.
        functions: List of available functions.
        test: The function calling test prompt.

    Returns:
        An OutputFile with the generated function call.
    """
    prompt = prompt_builder(functions, test)
    ids_list = model.encode(prompt)[0].tolist()
    functions_names = [function.name for function in functions]

    generated_so_far = get_function_name(
        model, vocab, functions_names, ids_list
    )
    function = find_function_name(functions, generated_so_far)

    if function is None:
        raise SystemExit("ERROR: Function not found")

    fixed_ids = model.encode('", "parameters": {')[0].tolist()
    ids_list += fixed_ids
    params: dict[str, Any] = {}
    value: float | str = 0

    for i, (param_name, param_info) in enumerate(
        function.parameters.items()
    ):
        prefix = (
            f'"{param_name}": ' if i == 0
            else f', "{param_name}": '
        )
        fixed_ids = model.encode(prefix)[0].tolist()
        ids_list += fixed_ids

        if param_info.type == "number":
            value, ids_list = generate_number_value(model, vocab, ids_list)

        elif param_info.type == "integer":
            value, ids_list = generate_number_value(model, vocab, ids_list)
            value = int(value)

        elif param_info.type == "string":
            value, ids_list = generate_string_value(model, vocab, ids_list)

        elif param_info.type == "boolean":
            value, ids_list = generate_boolean_value(model, vocab, ids_list)

        else:
            raise SystemExit(
                f"ERROR: Unknown parameter type: {param_info.type}"
            )
        params[param_name] = value

    return OutputFile(
        prompt=test.prompt, name=generated_so_far, parameters=params
    )
