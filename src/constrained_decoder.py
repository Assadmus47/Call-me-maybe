

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
