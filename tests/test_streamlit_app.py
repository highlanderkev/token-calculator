"""Tests for token_calculator.streamlit_app."""

from types import SimpleNamespace
from unittest.mock import patch

from token_calculator.streamlit_app import build_calculation, format_cost


@patch("token_calculator.streamlit_app.get_model_info")
@patch("token_calculator.streamlit_app.estimate_cost")
@patch("token_calculator.streamlit_app.count_tokens")
def test_build_calculation_includes_context_and_cost(
    mock_count_tokens,
    mock_estimate_cost,
    mock_get_model_info,
):
    mock_count_tokens.return_value = 42
    mock_estimate_cost.return_value = 0.12345678
    mock_get_model_info.return_value = SimpleNamespace(context_window=256)

    result = build_calculation("hello", model="gpt-4o", direction="input")

    assert result["model"] == "gpt-4o"
    assert result["direction"] == "input"
    assert result["token_count"] == 42
    assert result["estimated_cost_usd"] == 0.12345678
    assert result["context_window"] == 256
    assert result["remaining_context"] == 214


def test_format_cost_handles_missing_pricing():
    assert format_cost(None) == "N/A"


def test_format_cost_formats_float():
    assert format_cost(0.00021) == "$0.00021000"
