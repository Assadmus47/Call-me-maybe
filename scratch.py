from llm_sdk.llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
print("modèle chargé !")

prompt = "The capital of France is \""


ids = model.encode(prompt)
logits = model.get_logits_from_input_ids(ids[0].tolist())
max_id = logits.index(max(logits))
print(model.decode([max_id]))