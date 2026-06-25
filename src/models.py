from pydantic import BaseModel, Field
from typing import Any

class FunctionParameter(BaseModel):
    type: str


class Function(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    parameters: dict[str, FunctionParameter] = Field(...)
    returns: FunctionParameter = Field(...)


class FunctionCallingTest(BaseModel):
    prompt: str = Field(...)


class OutputFile(BaseModel):
    prompt: str = Field(...)
    name: str = Field(...)
    parameters: dict[str, Any] = Field(...)