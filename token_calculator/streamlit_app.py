"""Minimal Streamlit UI for manual token-calculator evaluation."""

from __future__ import annotations

from typing import Any, Optional

from token_calculator.calculator import count_tokens, estimate_cost
from token_calculator.models import get_model_info, list_models

__all__ = ["build_calculation", "format_cost", "main"]


def build_calculation(
    text: str,
    model: str = "gpt-4o",
    direction: str = "input",
) -> dict[str, Any]:
    """Return token, context, and cost details for the UI."""
    token_count = count_tokens(text, model=model)
    model_info = get_model_info(model)
    return {
        "model": model,
        "direction": direction,
        "token_count": token_count,
        "estimated_cost_usd": estimate_cost(token_count, model=model, direction=direction),
        "context_window": model_info.context_window,
        "remaining_context": max(model_info.context_window - token_count, 0),
    }


def format_cost(cost: Optional[float]) -> str:
    """Format a cost value for display."""
    return "N/A" if cost is None else f"${cost:.8f}"


def main() -> None:
    """Launch the Streamlit UI."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "streamlit is required for the UI. Install it with: "
            "pip install 'token-calculator[ui]'"
        ) from exc

    models = list_models()
    default_model_index = models.index("gpt-4o") if "gpt-4o" in models else 0

    st.set_page_config(page_title="token-calculator", layout="wide")
    st.title("token-calculator")
    st.caption("Minimal Streamlit UI for testing token counts and cost estimates.")

    with st.sidebar:
        model = st.selectbox("Model", models, index=default_model_index)
        direction = st.radio("Cost direction", ["input", "output"], horizontal=True)

    text = st.text_area(
        "Text",
        height=240,
        placeholder="Paste or type text to evaluate.",
    )

    if not text.strip():
        st.info("Enter text to calculate tokens and estimate cost.")
        return

    result = build_calculation(text=text, model=model, direction=direction)
    token_count = result["token_count"]
    context_window = result["context_window"]
    remaining_context = result["remaining_context"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tokens", f"{token_count:,}")
    col2.metric("Estimated cost", format_cost(result["estimated_cost_usd"]))
    col3.metric("Context window", f"{context_window:,}")
    col4.metric("Remaining context", f"{remaining_context:,}")

    st.subheader("Details")
    st.json(result)


if __name__ == "__main__":  # pragma: no cover
    main()
