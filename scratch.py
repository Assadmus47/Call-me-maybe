from llm_sdk.llm_sdk import Small_LLM_Model
import json

from src.file_parser import parse_function_file, parse_calling_function_file
from src.constrained_decoder import *
from src.prompt_builder import prompt_builder

functions = parse_function_file("data/input/functions_definition.json")
tests = parse_calling_function_file("data/input/function_calling_tests.json")

prompt = prompt_builder(functions, tests[1])
print(prompt)


model = Small_LLM_Model()
print("modèle chargé !")

ids = model.encode(prompt)
# logits = model.get_logits_from_input_ids(ids[0].tolist())
# max_id = logits.index(max(logits))
# print(model.decode([max_id]))
# path = model.get_path_to_vocab_file()
# print(path)

# id_to_token = {v: k for k, v in vocab.items()}
# token_string = id_to_token[max_id]
# print(token_string)
# max_score = max(logits)
# max_id = logits.index(max_score)

path = model.get_path_to_vocab_file()
with open(path, "r") as f:
    vocab = json.load(f)

logits = model.get_logits_from_input_ids(ids[0].tolist())


function_names = ["fn_add_numbers", "fn_greet"]
ids_list = ids[0].tolist()
generated_so_far = ""

while (generated_so_far not in function_names):
    logits = model.get_logits_from_input_ids(ids_list)
    
    logits = find_valide_token(logits, vocab, function_names, generated_so_far)

    max_id = logits.index(max(logits))
    generated_so_far += model.decode([max_id])
    ids_list.append(max_id)

function = find_function_name(functions, generated_so_far)
if function is None:
    raise SystemExit("ERROR: Function not found")

print(repr(generated_so_far))
fixed_ids = model.encode('", "parameters": {')[0].tolist()
ids_list += fixed_ids

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
    
    print(f"{param_name}: {value}")
