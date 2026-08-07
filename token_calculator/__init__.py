"""
token_calculator — LLM token counting and cost estimation plugin.

Quick start::

    from token_calculator import count_tokens, estimate_cost

    n = count_tokens("Hello, world!", model="gpt-4o")
    cost = estimate_cost(n, model="gpt-4o")
"""

from token_calculator.calculator import count_tokens, estimate_cost
from token_calculator.models import get_model_info, list_models
from token_calculator.plugin import (
    OPENAI_TOOL_SPEC,
    TokenCalculatorTool,
    token_calculator_callable,
)

__all__ = [
    "count_tokens",
    "estimate_cost",
    "get_model_info",
    "list_models",
    "OPENAI_TOOL_SPEC",
    "TokenCalculatorTool",
    "token_calculator_callable",
]
