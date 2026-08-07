"""
Core token-counting and cost-estimation logic.

Supports two tokenizer backends:
  * tiktoken  — used for OpenAI models (fast, offline).
  * transformers — used for Hugging Face-hosted models (requires the
    ``transformers`` extra and network access on first use).

The backend is selected automatically based on the model's ``tokenizer``
field in the registry (e.g. ``"tiktoken:cl100k_base"`` or
``"transformers:meta-llama/Meta-Llama-3-8B"``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from token_calculator.models import ModelInfo, get_model_info

__all__ = ["count_tokens", "estimate_cost"]


# ── Tokenizer helpers ──────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def _get_tiktoken_encoder(encoding_name: str):
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tiktoken is required for OpenAI models. "
            "Install it with: pip install tiktoken"
        ) from exc
    return tiktoken.get_encoding(encoding_name)


@lru_cache(maxsize=16)
def _get_transformers_tokenizer(hf_model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for non-OpenAI models. "
            "Install it with: pip install token-calculator[transformers]"
        ) from exc
    return AutoTokenizer.from_pretrained(hf_model_id)


def _count_with_tiktoken(text: str, encoding_name: str) -> int:
    enc = _get_tiktoken_encoder(encoding_name)
    return len(enc.encode(text))


def _count_with_transformers(text: str, hf_model_id: str) -> int:
    tokenizer = _get_transformers_tokenizer(hf_model_id)
    return len(tokenizer.encode(text, add_special_tokens=False))


# ── Public API ─────────────────────────────────────────────────────────────


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in *text* for *model*.

    Parameters
    ----------
    text:
        The string to tokenize.
    model:
        A model name registered in :mod:`token_calculator.models`.

    Returns
    -------
    int
        Token count.

    Raises
    ------
    KeyError
        If *model* is not in the registry.
    ImportError
        If the required tokenizer library is not installed.
    """
    info: ModelInfo = get_model_info(model)
    backend, identifier = info.tokenizer.split(":", 1)

    if backend == "tiktoken":
        return _count_with_tiktoken(text, identifier)
    if backend == "transformers":
        return _count_with_transformers(text, identifier)

    raise ValueError(f"Unknown tokenizer backend '{backend}' for model '{model}'")


def estimate_cost(
    tokens: int,
    model: str = "gpt-4o",
    direction: str = "input",
) -> Optional[float]:
    """Estimate the USD cost for *tokens* tokens with *model*.

    Parameters
    ----------
    tokens:
        Number of tokens.
    model:
        A model name registered in :mod:`token_calculator.models`.
    direction:
        ``"input"`` (default) or ``"output"``.

    Returns
    -------
    float or None
        Estimated cost in USD, or ``None`` when pricing is not available.

    Raises
    ------
    KeyError
        If *model* is not in the registry.
    ValueError
        If *direction* is not ``"input"`` or ``"output"``.
    """
    if direction not in ("input", "output"):
        raise ValueError(f"direction must be 'input' or 'output', got '{direction}'")

    info: ModelInfo = get_model_info(model)
    cost_per_1k = (
        info.input_cost_per_1k if direction == "input" else info.output_cost_per_1k
    )

    if cost_per_1k is None:
        return None

    return round(tokens / 1000 * cost_per_1k, 8)
