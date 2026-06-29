from typing import Any
from pydantic import BaseModel, Field


class FunctionParameter(BaseModel):
    """Represents a function parameter with its type."""

    type: str


class Function(BaseModel):
    """Represents a callable function with its schema."""
    model_config = {"extra": "forbid"}
    name: str = Field(...)
    description: str = Field(...)
    parameters: dict[str, FunctionParameter] = Field(...)
    returns: FunctionParameter = Field(...)


class FunctionCallingTest(BaseModel):
    """Represents a natural language prompt to process."""
    model_config = {"extra": "forbid"}
    prompt: str = Field(...)


class OutputFile(BaseModel):
    """Represents a generated function call result."""

    prompt: str = Field(...)
    name: str = Field(...)
    parameters: dict[str, Any] = Field(...)
