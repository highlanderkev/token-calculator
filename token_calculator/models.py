"""
Model registry for token-calculator.

Each entry stores:
  - context_window: maximum tokens supported by the model
  - input_cost_per_1k: USD per 1 000 input tokens (None = unknown / free)
  - output_cost_per_1k: USD per 1 000 output tokens (None = unknown / free)
  - tokenizer: "tiktoken:<encoding>" or "transformers:<hf-model-id>"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["ModelInfo", "MODEL_REGISTRY", "get_model_info", "list_models"]


@dataclass(frozen=True)
class ModelInfo:
    name: str
    context_window: int
    input_cost_per_1k: Optional[float]
    output_cost_per_1k: Optional[float]
    tokenizer: str


MODEL_REGISTRY: dict[str, ModelInfo] = {
    # ── OpenAI ─────────────────────────────────────────────────────────────
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        context_window=128_000,
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.015,
        tokenizer="tiktoken:o200k_base",
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        context_window=128_000,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        tokenizer="tiktoken:o200k_base",
    ),
    "gpt-4-turbo": ModelInfo(
        name="gpt-4-turbo",
        context_window=128_000,
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        tokenizer="tiktoken:cl100k_base",
    ),
    "gpt-4": ModelInfo(
        name="gpt-4",
        context_window=8_192,
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
        tokenizer="tiktoken:cl100k_base",
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        context_window=16_385,
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        tokenizer="tiktoken:cl100k_base",
    ),
    # ── Anthropic Claude ───────────────────────────────────────────────────
    # NOTE: Xenova/claude-tokenizer is a community approximation; token counts
    # for Claude models may differ slightly from Anthropic's internal tokenizer.
    "claude-3-5-sonnet": ModelInfo(
        name="claude-3-5-sonnet",
        context_window=200_000,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        tokenizer="transformers:Xenova/claude-tokenizer",
    ),
    "claude-3-opus": ModelInfo(
        name="claude-3-opus",
        context_window=200_000,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        tokenizer="transformers:Xenova/claude-tokenizer",
    ),
    "claude-3-sonnet": ModelInfo(
        name="claude-3-sonnet",
        context_window=200_000,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        tokenizer="transformers:Xenova/claude-tokenizer",
    ),
    "claude-3-haiku": ModelInfo(
        name="claude-3-haiku",
        context_window=200_000,
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        tokenizer="transformers:Xenova/claude-tokenizer",
    ),
    # ── Google Gemini ──────────────────────────────────────────────────────
    "gemini-1.5-pro": ModelInfo(
        name="gemini-1.5-pro",
        context_window=2_097_152,
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.005,
        tokenizer="tiktoken:cl100k_base",  # approximation
    ),
    "gemini-1.5-flash": ModelInfo(
        name="gemini-1.5-flash",
        context_window=1_048_576,
        input_cost_per_1k=0.000075,
        output_cost_per_1k=0.0003,
        tokenizer="tiktoken:cl100k_base",  # approximation
    ),
    # ── Meta Llama ─────────────────────────────────────────────────────────
    "llama-3-8b": ModelInfo(
        name="llama-3-8b",
        context_window=8_192,
        input_cost_per_1k=None,
        output_cost_per_1k=None,
        tokenizer="transformers:meta-llama/Meta-Llama-3-8B",
    ),
    "llama-3-70b": ModelInfo(
        name="llama-3-70b",
        context_window=8_192,
        input_cost_per_1k=None,
        output_cost_per_1k=None,
        tokenizer="transformers:meta-llama/Meta-Llama-3-70B",
    ),
    # ── Mistral ────────────────────────────────────────────────────────────
    "mistral-7b": ModelInfo(
        name="mistral-7b",
        context_window=32_768,
        input_cost_per_1k=None,
        output_cost_per_1k=None,
        tokenizer="transformers:mistralai/Mistral-7B-v0.1",
    ),
}


def get_model_info(model: str) -> ModelInfo:
    """Return :class:`ModelInfo` for *model*, raising ``KeyError`` if unknown."""
    if model not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model}'. Available models: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model]


def list_models() -> list[str]:
    """Return a sorted list of registered model names."""
    return sorted(MODEL_REGISTRY)
