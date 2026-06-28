from typing import Any
from src.models import Function


def find_valide_token(
    logits: list[float],
    vocab: dict[str, Any],
    function_names: list[str],
    generated_so_far: str,
) -> list[float]:
    """Filter logits to only allow tokens that continue a valid function name.

    Args:
        logits: The logits from the LLM.
        vocab: The vocabulary dictionary.
        function_names: List of valid function names.
        generated_so_far: The text generated so far.

    Returns:
        Modified logits with invalid tokens set to -inf.
    """
    id_to_token = {v: k for k, v in vocab.items()}
    for token_id, score in enumerate(logits):
        if token_id not in id_to_token:
            logits[token_id] = float('-inf')
            continue
        if not any(
            fn.startswith(generated_so_far + id_to_token[token_id])
            for fn in function_names
        ):
            logits[token_id] = float('-inf')
    return logits


def find_valid_number_tokens(
    logits: list[float],
    vocab: dict[str, Any],
) -> list[float]:
    """Filter logits to only allow tokens that form a valid number.

    Args:
        logits: The logits from the LLM.
        vocab: The vocabulary dictionary.

    Returns:
        Modified logits with invalid tokens set to -inf.
    """
    id_to_token = {v: k for k, v in vocab.items()}
    for token_id, score in enumerate(logits):
        if token_id not in id_to_token:
            logits[token_id] = float('-inf')
            continue
        token = id_to_token[token_id]
        if token in [".", "-", ",", "}"]:
            continue
        if not token.replace(".", "").replace("-", "").isdigit():
            logits[token_id] = float('-inf')
    return logits


def find_valid_boolean_tokens(
    logits: list[float],
    vocab: dict[str, Any],
) -> list[float]:
    """Filter logits to only allow tokens that form true or false.

    Args:
        logits: The logits from the LLM.
        vocab: The vocabulary dictionary.

    Returns:
        Modified logits with invalid tokens set to -inf.
    """
    id_to_token = {v: k for k, v in vocab.items()}
    for token_id, score in enumerate(logits):
        if token_id not in id_to_token:
            logits[token_id] = float('-inf')
            continue
        token = id_to_token[token_id]
        if not (
            "true".startswith(token) or "false".startswith(token)
        ):
            logits[token_id] = float('-inf')
    return logits


def find_valid_string_tokens(
    logits: list[float],
    vocab: dict[str, Any],
) -> list[float]:
    """Filter logits to only allow valid string tokens.

    Args:
        logits: The logits from the LLM.
        vocab: The vocabulary dictionary.

    Returns:
        Modified logits with invalid tokens set to -inf.
    """
    id_to_token = {v: k for k, v in vocab.items()}
    for token_id, score in enumerate(logits):
        if token_id not in id_to_token:
            logits[token_id] = float('-inf')
    return logits


def find_function_name(
    functions: list[Function],
    name_target: str,
) -> Function | None:
    """Find a function by name in a list of functions.

    Args:
        functions: List of available functions.
        name_target: The function name to find.

    Returns:
        The matching Function or None.
    """
    for func in functions:
        if func.name == name_target:
            return func
    return None


def generate_number_value(
    model: Any,
    vocab: dict[str, Any],
    ids_list: list[int],
) -> tuple[float, list[int]]:
    """Generate a number value using constrained decoding.

    Args:
        model: The LLM model instance.
        vocab: The vocabulary dictionary.
        ids_list: Current list of token IDs.

    Returns:
        A tuple of (float value, updated ids_list).
    """
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
        result = float(value)
    except ValueError:
        raise SystemExit("ERROR: Could not generate number value")
    return result, ids_list


def generate_boolean_value(
    model: Any,
    vocab: dict[str, Any],
    ids_list: list[int],
) -> tuple[str, list[int]]:
    """Generate a boolean value using constrained decoding.

    Args:
        model: The LLM model instance.
        vocab: The vocabulary dictionary.
        ids_list: Current list of token IDs.

    Returns:
        A tuple of (bool string value, updated ids_list).
    """
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


def generate_string_value(
    model: Any,
    vocab: dict[str, Any],
    ids_list: list[int],
) -> tuple[str, list[int]]:
    """Generate a string value using constrained decoding.

    Args:
        model: The LLM model instance.
        vocab: The vocabulary dictionary.
        ids_list: Current list of token IDs.

    Returns:
        A tuple of (string value, updated ids_list).
    """
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
