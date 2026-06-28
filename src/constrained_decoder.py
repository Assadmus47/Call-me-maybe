from src.models import Function
from typing import Any


def find_valide_token(logits: list[float], vocab: dict[str, float], function_names: list[str], generated_so_far: str) -> list[float]:
    id_to_token = {v: k for k, v in vocab.items()}
    for token_id, score in enumerate(logits):
        if token_id not in id_to_token:
            logits[token_id] = float('-inf')
            continue
        if not any(function_name.startswith(
            generated_so_far + id_to_token[token_id]) for function_name in function_names
        ):
            logits[token_id] = float('-inf')
    return logits


def find_valid_number_tokens(logits: list[float], vocab: dict[str, float]) -> list[float]:
    id_to_token = {v: k for k, v in vocab.items()}
    for token_id, score in enumerate(logits):
        if token_id not in id_to_token:
            logits[token_id] = float('-inf')
            continue
        if id_to_token[token_id] in [".", "-", ",", "}"]:
            continue
        if not id_to_token[token_id].replace(".", "").replace("-", "").isdigit():
            logits[token_id] = float('-inf')

    return logits


def find_valid_boolean_tokens(logits: list[float], vocab: dict[str, float]) -> list[float]:
    id_to_token = {v: k for k, v in vocab.items()}
    for token_id, score in enumerate(logits):
        if token_id not in id_to_token:
            logits[token_id] = float('-inf')
            continue
        if not ("true".startswith(id_to_token[token_id]) or "false".startswith(id_to_token[token_id])):
            logits[token_id] = float('-inf')

    return logits


def find_function_name(functions: list[Function], name_target: str) -> Function | None:
    for func in functions:
        if func.name == name_target:
            return func
    return None


def find_valid_string_tokens(logits: list[float], vocab: dict[str, float]) -> list[float]:
    id_to_token = {v: k for k, v in vocab.items()}
    for token_id, score in enumerate(logits):
        if token_id not in id_to_token:
            logits[token_id] = float('-inf')
    return logits


def generate_number_value(model: Any, vocab: dict[str, float], ids_list: list[int]) -> tuple[float, list[int]]:
    last_token = ""
    value = ""
    while last_token not in [",", "}"]:
        logits = model.get_logits_from_input_ids(ids_list)
        logits = find_valid_number_tokens(logits, vocab)
        max_id = logits.index(max(logits))
        last_token = model.decode([max_id])
        ids_list.append(max_id)
        if last_token not in [",", "}"]:
            value += last_token
    try:
        value = float(value)
    except ValueError:
        raise SystemExit("ERROR: Could not generate number value")
    return value, ids_list


def generate_boolean_value(model: Any, vocab: dict[str, float], ids_list: list[int]) -> tuple[str, list[int]]:
    last_token = ""
    value = ""
    while value not in ["true", "false"]:
        logits = model.get_logits_from_input_ids(ids_list)
        logits = find_valid_boolean_tokens(logits, vocab)
        max_id = logits.index(max(logits))
        last_token = model.decode([max_id])
        ids_list.append(max_id)
        if last_token not in [",", "}"]:
            value += last_token
    return value, ids_list


def generate_string_value(model: Any, vocab: dict[str, float], ids_list: list[int]) -> tuple[str, list[int]]:
    value = ""
    found_first_quote = False
    while True:
        logits = model.get_logits_from_input_ids(ids_list)
        logits = find_valid_string_tokens(logits, vocab)
        max_id = logits.index(max(logits))
        last_token = model.decode([max_id])
        ids_list.append(max_id)
        
        if '"' in last_token:
            if found_first_quote:
                break
            found_first_quote = True
        else:
            if found_first_quote:
                value += last_token
    return value, ids_list
