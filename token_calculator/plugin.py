"""
Plugin adapters for agentic AI systems.

Three integration points are provided:

1. ``OPENAI_TOOL_SPEC`` — JSON schema dict that can be passed directly to
   OpenAI's ``tools`` parameter (Assistants API, Chat Completions, etc.).

2. ``TokenCalculatorTool`` — a LangChain ``BaseTool`` subclass (requires the
   ``langchain`` extra).  Falls back gracefully to a plain-Python class when
   LangChain is not installed so that the module always imports cleanly.

3. ``token_calculator_callable`` — a plain Python function compatible with
   CrewAI, AutoGen, and any framework that accepts ``Callable[[str], str]``.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from token_calculator.calculator import count_tokens, estimate_cost
from token_calculator.models import list_models

__all__ = [
    "OPENAI_TOOL_SPEC",
    "TokenCalculatorTool",
    "token_calculator_callable",
]

# ── 1. OpenAI / OpenAI-compatible tool spec ────────────────────────────────

# NOTE: OPENAI_TOOL_SPEC is built once at import time. The embedded model list
# reflects the registry state at that moment and will not update dynamically.
OPENAI_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "token_calculator",
        "description": (
            "Count the number of tokens in a text string for a given LLM model "
            "and optionally estimate the USD cost."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to tokenize.",
                },
                "model": {
                    "type": "string",
                    "description": (
                        "The model name to use for tokenization. "
                        f"Available models: {list_models()}."
                    ),
                    "default": "gpt-4o",
                },
                "direction": {
                    "type": "string",
                    "enum": ["input", "output"],
                    "description": "Whether to estimate input or output token cost.",
                    "default": "input",
                },
            },
            "required": ["text"],
        },
    },
}


# ── 2. LangChain tool ──────────────────────────────────────────────────────

def _build_langchain_tool():
    """Return a LangChain BaseTool subclass, or a stub if not installed."""
    try:
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, Field

        class _TokenInput(BaseModel):
            text: str = Field(..., description="Text to tokenize")
            model: str = Field("gpt-4o", description="Model name")
            direction: str = Field("input", description="'input' or 'output'")

        class _TokenCalculatorTool(BaseTool):
            name: str = "token_calculator"
            description: str = (
                "Count tokens in text for a given LLM model and estimate USD cost."
            )
            args_schema: Type[BaseModel] = _TokenInput

            def _run(
                self,
                text: str,
                model: str = "gpt-4o",
                direction: str = "input",
                **kwargs: Any,
            ) -> str:
                return _invoke(text, model, direction)

            async def _arun(
                self,
                text: str,
                model: str = "gpt-4o",
                direction: str = "input",
                **kwargs: Any,
            ) -> str:  # pragma: no cover
                import asyncio

                return await asyncio.to_thread(self._run, text, model, direction)

        return _TokenCalculatorTool

    except ImportError:
        # LangChain not installed — return a plain stub that still works as a
        # callable so imports never fail.
        class _StubTool:  # type: ignore[no-redef]
            """Stub returned when langchain-core is not installed."""

            name = "token_calculator"
            description = (
                "Install langchain-core to use the LangChain tool adapter: "
                "pip install 'token-calculator[langchain]'"
            )

            def run(self, text: str, model: str = "gpt-4o", direction: str = "input") -> str:
                return _invoke(text, model, direction)

            __call__ = run

        return _StubTool


TokenCalculatorTool = _build_langchain_tool()


# ── 3. Plain callable (CrewAI / AutoGen / generic) ─────────────────────────

def _invoke(text: str, model: str = "gpt-4o", direction: str = "input") -> str:
    """Shared invocation logic — returns a JSON string result."""
    n = count_tokens(text, model=model)
    cost: Optional[float] = estimate_cost(n, model=model, direction=direction)
    return json.dumps(
        {
            "model": model,
            "token_count": n,
            "direction": direction,
            "estimated_cost_usd": cost,
        }
    )


def token_calculator_callable(
    text: str,
    model: str = "gpt-4o",
    direction: str = "input",
) -> str:
    """Count tokens and estimate cost, returning a JSON string.

    Suitable for use with CrewAI ``Tool``, AutoGen ``FunctionTool``, or any
    framework that expects a ``Callable[[str, ...], str]``.

    Parameters
    ----------
    text:
        The text to tokenize.
    model:
        Model name (see :func:`token_calculator.list_models`).
    direction:
        ``"input"`` or ``"output"``.

    Returns
    -------
    str
        JSON-encoded result with ``token_count`` and ``estimated_cost_usd``.
    """
    return _invoke(text, model, direction)
