"""Tests for token_calculator.streamlit_app."""

from unittest.mock import patch

from token_calculator.streamlit_app import build_calculation, format_cost


@patch("token_calculator.streamlit_app.count_tokens")
def test_build_calculation_includes_context_and_cost(mock_count_tokens):
    mock_count_tokens.return_value = 42

    result = build_calculation("hello", model="gpt-4o", direction="input")

    assert result["model"] == "gpt-4o"
    assert result["direction"] == "input"
    assert result["token_count"] == 42
    assert result["estimated_cost_usd"] == 0.00021
    assert result["context_window"] == 128_000
    assert result["remaining_context"] == 127_958


def test_format_cost_handles_missing_pricing():
    assert format_cost(None) == "N/A"


def test_format_cost_formats_float():
    assert format_cost(0.00021) == "$0.00021000"
