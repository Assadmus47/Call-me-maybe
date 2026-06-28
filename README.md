*This project has been created as part of the 42 curriculum by mkacemi.*

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

Call Me Maybe is a function calling system that translates natural language prompts into structured JSON function calls using a small Large Language Model (Qwen3-0.6B). Instead of relying on the model to spontaneously produce valid JSON, the system uses **constrained decoding** — a technique that guides token generation step by step to guarantee 100% valid and schema-compliant output.

Given a prompt like `"What is the sum of 2 and 3?"`, the system produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Instructions

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone <your-repo-url>
cd call-me-maybe
uv sync
```

Copy the provided `llm_sdk/` directory at the root of the project.

### Running the program

```bash
# With default paths
uv run python -m src

# With custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Makefile commands

```bash
make install     # Install dependencies
make run         # Run the program
make debug       # Run in debug mode (pdb)
make clean       # Remove caches
make lint        # Run flake8 + mypy
make lint-strict # Run flake8 + mypy --strict
```

## Algorithm Explanation

The constrained decoding pipeline works as follows:

### 1. Prompt Construction
A prompt is built combining the available functions (name, description, parameters) and the user's natural language request. This gives the LLM the context it needs to select the right function.

### 2. Function Name Generation
The LLM generates the function name token by token. At each step:
- The model produces logits (scores) for all 151,936 tokens in its vocabulary
- Only tokens that extend a valid function name prefix are kept
- All other tokens are masked to `-inf` (impossible to select)
- The highest-scoring valid token is chosen and appended

This guarantees the generated name is always one of the available functions.

### 3. Parameter Value Generation
For each parameter, the system adds the parameter name as fixed context and constrains token generation based on the parameter type:
- **number**: only digits, `.`, and `-` are allowed; stops on `,` or `}`
- **string**: generates freely until the closing `"` is detected
- **boolean**: only tokens that are prefixes of `"true"` or `"false"` are allowed

### 4. Output
Results are assembled into a structured JSON file matching the required schema.

## Design Decisions

- **No external structured generation libraries**: outlines, dspy, and similar libraries are forbidden. The constrained decoding is implemented from scratch using only numpy, pydantic, and the provided `llm_sdk`.
- **Pydantic for all models**: `FunctionParameter`, `Function`, `FunctionCallingTest`, and `OutputFile` are all validated with Pydantic.
- **Token-level masking**: instead of post-processing the output, invalid tokens are masked before selection — this makes invalid output structurally impossible.
- **uv for dependency management**: chosen for speed and reproducibility via `uv.lock`.
- **Separation of concerns**: parsing, prompt building, constrained decoding, and generation are each in their own module.

## Performance Analysis

- **JSON validity**: 100% — constrained decoding makes malformed output impossible
- **Function selection accuracy**: ~90%+ on standard prompts with clear intent
- **Speed**: approximately 30-60 seconds per prompt on CPU (Qwen3-0.6B is small but still requires computation per token)
- **Reliability**: the system never crashes on valid input; all edge cases (unknown types, missing files, invalid JSON) are handled with clear error messages

## Challenges Faced

- **Token boundary mismatch**: the tokenizer does not split text word by word. A single token can span multiple characters or contain leading spaces (e.g. `" fn"`). The prefix-matching logic in `find_valide_token` had to account for this.
- **String termination**: detecting the closing `"` of a string value was tricky because the tokenizer sometimes includes the quote in a multi-character token. A `found_first_quote` flag was used to correctly detect the second quote.
- **mypy + llm_sdk**: the provided SDK has type errors internally. A `mypy.ini` was added to exclude it from type checking without affecting the project's own code quality.
- **Number conversion**: generated number tokens are strings by default; explicit `float()` conversion with error handling was added.

## Testing Strategy

- Manual testing with `data/input/function_calling_tests.json` covering:
  - Number parameters (`fn_add_numbers`)
  - String parameters (`fn_greet`)
  - Multiple prompts in sequence
- Edge case validation:
  - Missing input files → clear error message
  - Invalid JSON in input → clear error message
  - Unknown parameter type → clear error message
- Lint and type checking with `make lint` and `make lint-strict` passing with zero errors

## Example Usage

```bash
# Default run
make run

# Custom input
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Example output (`data/output/function_calls.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

## Resources

- [Qwen3 Model — HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained Decoding — Outlines library (reference)](https://github.com/outlines-dev/outlines)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [mypy Documentation](https://mypy.readthedocs.io/)

### AI Usage

Claude (Anthropic) was used throughout this project as a teaching assistant and pair programming tool. Specifically:
- Explaining theoretical concepts (tokenization, constrained decoding, LLM internals)
- Guiding the implementation step by step while letting the student write the code
- Reviewing and cleaning up code for flake8/mypy compliance
- Helping debug edge cases (string termination, token boundary issues)

All code was written and understood by the student. Claude was never used to generate complete solutions without understanding.
