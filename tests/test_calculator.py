"""Tests for token_calculator.calculator and token_calculator.models."""

from unittest.mock import MagicMock, patch

import pytest

from token_calculator.models import get_model_info, list_models


# ── model registry ─────────────────────────────────────────────────────────

def test_list_models_returns_sorted_list():
    models = list_models()
    assert isinstance(models, list)
    assert models == sorted(models)
    assert "gpt-4o" in models


def test_get_model_info_known():
    info = get_model_info("gpt-4o")
    assert info.name == "gpt-4o"
    assert info.context_window == 128_000
    assert info.input_cost_per_1k > 0
    assert info.tokenizer.startswith("tiktoken:")


def test_get_model_info_unknown():
    with pytest.raises(KeyError, match="Unknown model"):
        get_model_info("not-a-model")


# ── count_tokens ───────────────────────────────────────────────────────────

def _fake_tiktoken_encoder(n_tokens: int):
    enc = MagicMock()
    enc.encode.return_value = list(range(n_tokens))
    return enc


@patch("token_calculator.calculator._get_tiktoken_encoder")
def test_count_tokens_gpt4o(mock_enc):
    mock_enc.return_value = _fake_tiktoken_encoder(4)
    from token_calculator.calculator import count_tokens
    n = count_tokens("Hello, world!", model="gpt-4o")
    assert n == 4


@patch("token_calculator.calculator._get_tiktoken_encoder")
def test_count_tokens_empty_string(mock_enc):
    mock_enc.return_value = _fake_tiktoken_encoder(0)
    from token_calculator.calculator import count_tokens
    n = count_tokens("", model="gpt-4o")
    assert n == 0


@patch("token_calculator.calculator._get_tiktoken_encoder")
def test_count_tokens_longer_text(mock_enc):
    mock_enc.return_value = _fake_tiktoken_encoder(90)
    from token_calculator.calculator import count_tokens
    text = "The quick brown fox jumps over the lazy dog. " * 10
    n = count_tokens(text, model="gpt-3.5-turbo")
    assert n > 50


def test_count_tokens_unknown_model():
    from token_calculator.calculator import count_tokens
    with pytest.raises(KeyError):
        count_tokens("hello", model="not-a-real-model")


# ── estimate_cost ──────────────────────────────────────────────────────────

def test_estimate_cost_input():
    from token_calculator.calculator import estimate_cost
    cost = estimate_cost(1000, model="gpt-4o", direction="input")
    assert cost is not None
    assert cost == pytest.approx(0.005, rel=1e-3)


def test_estimate_cost_output():
    from token_calculator.calculator import estimate_cost
    cost = estimate_cost(1000, model="gpt-4o", direction="output")
    assert cost is not None
    assert cost == pytest.approx(0.015, rel=1e-3)


def test_estimate_cost_no_pricing():
    from token_calculator.calculator import estimate_cost
    cost = estimate_cost(1000, model="llama-3-8b")
    assert cost is None


def test_estimate_cost_bad_direction():
    from token_calculator.calculator import estimate_cost
    with pytest.raises(ValueError, match="direction"):
        estimate_cost(100, direction="sideways")


def test_estimate_cost_zero_tokens():
    from token_calculator.calculator import estimate_cost
    cost = estimate_cost(0, model="gpt-4o")
    assert cost == 0.0
