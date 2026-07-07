"""DSPy-free optimizer helpers.

Pure helpers used by the COPRO optimizer and its tests, kept here so importing
them does not pull in ``dspy`` (whose avatar modules emit import-time
DeprecationWarnings). The DSPy-dependent optimizer path remains in
``optimize_dspy.py``.
"""


def size_penalized_utility(score: float, api_prompt_ratio: float, penalty: float) -> float:
    """Penalize COPRO utility only when the instruction grew beyond the baseline.

    ratio <= 1.0 (prompt shrank or stayed same) returns the raw score unchanged;
    growth reduces utility linearly. This steers COPRO candidate selection only
    and is NEVER applied to the human-facing baseline_mean/optimized_mean numbers.
    """
    return score - penalty * max(0.0, api_prompt_ratio - 1.0)


def _to_pydantic_ai_model_id(model_id: str) -> str:
    """Translate optimize-only DSPy model ids to Pydantic AI ids for eval scoring."""
    # Branch boundary: already in Pydantic AI provider-prefixed format.
    if ":" in model_id:
        return model_id
    slash_provider, has_slash, slash_model = model_id.partition("/")
    # Branch boundary: DSPy/LiteLLM uses provider/model while Pydantic AI uses provider:model.
    if has_slash and slash_provider in {"anthropic", "cohere", "groq", "mistral", "openai", "openrouter", "xai"}:
        return f"{slash_provider}:{slash_model}"
    if has_slash and slash_provider in {"gemini", "google"}:
        return f"google:{slash_model}"
    # Branch boundary: common provider short ids used by optimizer flags.
    if model_id.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai:" + model_id
    if model_id.startswith("claude-"):
        return "anthropic:" + model_id
    if model_id.startswith("gemini-"):
        return "google:" + model_id
    return model_id
