from src.models import Function, FunctionCallingTest


def prompt_builder(functions_definitions: list[Function], function_calling_tests: FunctionCallingTest) -> str:
    functions_definition = []
    for func in functions_definitions:
        parameters = ""
        for param_name, param_info in func.parameters.items():
            parameters += f"{param_name} ({param_info.type}),"
        parameters = parameters.rstrip(",")
        functions_definition.append(
            f"{func.name}: {func.description}. "
            f"Parametres: {parameters}. Returns: {func.returns.type}"
        )
    functions_details = "\n".join(functions_definition)

    return f"""Tu as ces fonctions disponibles :
        [{functions_details}]
        Question : [{function_calling_tests.prompt}]
        Réponds en JSON : {{\"name": \""""
