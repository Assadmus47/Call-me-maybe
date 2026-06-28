from src.models import Function, FunctionCallingTest


def prompt_builder(
    functions_definitions: list[Function],
    function_calling_tests: FunctionCallingTest,
) -> str:
    """Build a prompt for the LLM from functions and a test prompt.

    Args:
        functions_definitions: List of available functions.
        function_calling_tests: The test prompt to process.

    Returns:
        A formatted prompt string for the LLM.
    """
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

    return (
        f"Tu as ces fonctions disponibles :\n"
        f"        [{functions_details}]\n"
        f"        Question : [{function_calling_tests.prompt}]\n"
        f'        Réponds en JSON : {{"name": "'
    )
