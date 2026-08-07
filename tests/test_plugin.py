"""Tests for token_calculator.plugin."""

import json
from unittest.mock import MagicMock, patch

import pytest

from token_calculator.plugin import (
    OPENAI_TOOL_SPEC,
    token_calculator_callable,
    TokenCalculatorTool,
)


def _fake_tiktoken_encoder(n_tokens: int):
    enc = MagicMock()
    enc.encode.return_value = list(range(n_tokens))
    return enc


# ── OPENAI_TOOL_SPEC ───────────────────────────────────────────────────────

def test_openai_tool_spec_structure():
    assert OPENAI_TOOL_SPEC["type"] == "function"
    fn = OPENAI_TOOL_SPEC["function"]
    assert fn["name"] == "token_calculator"
    params = fn["parameters"]
    assert "text" in params["properties"]
    assert "text" in params["required"]


# ── token_calculator_callable ──────────────────────────────────────────────

@patch("token_calculator.calculator._get_tiktoken_encoder")
def test_callable_returns_json(mock_enc):
    mock_enc.return_value = _fake_tiktoken_encoder(4)
    result = token_calculator_callable("Hello, world!", model="gpt-4o")
    data = json.loads(result)
    assert data["model"] == "gpt-4o"
    assert data["token_count"] == 4
    assert "estimated_cost_usd" in data


@patch("token_calculator.calculator._get_tiktoken_encoder")
def test_callable_output_direction(mock_enc):
    mock_enc.return_value = _fake_tiktoken_encoder(5)
    result = token_calculator_callable("test", model="gpt-4o", direction="output")
    data = json.loads(result)
    assert data["direction"] == "output"
    assert data["estimated_cost_usd"] is not None


@patch("token_calculator.calculator._get_tiktoken_encoder")
def test_callable_no_pricing_model(mock_enc):
    # llama uses transformers backend — patch it too
    fake_tok = MagicMock()
    fake_tok.encode.return_value = [1, 2, 3]
    with patch("token_calculator.calculator._get_transformers_tokenizer", return_value=fake_tok):
        result = token_calculator_callable("test", model="llama-3-8b")
    data = json.loads(result)
    assert data["estimated_cost_usd"] is None


# ── TokenCalculatorTool ────────────────────────────────────────────────────

def test_tool_has_name_and_description():
    tool = TokenCalculatorTool()
    assert hasattr(tool, "name")
    assert hasattr(tool, "description")


@patch("token_calculator.calculator._get_tiktoken_encoder")
def test_stub_tool_run_when_no_langchain(mock_enc):
    mock_enc.return_value = _fake_tiktoken_encoder(3)
    tool = TokenCalculatorTool()
    if hasattr(tool, "_run"):
        result = tool._run("Hello", model="gpt-4o", direction="input")
    else:
        result = tool.run("Hello", model="gpt-4o", direction="input")
    data = json.loads(result)
    assert data["token_count"] == 3
