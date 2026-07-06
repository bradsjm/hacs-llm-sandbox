import pytest
from llm_sandbox_evals.optimize_dspy import _to_pydantic_ai_model_id


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        pytest.param("openai:gpt-4o-mini", "openai:gpt-4o-mini", id="already-pydantic-ai"),
        pytest.param("openai/gpt-4o-mini", "openai:gpt-4o-mini", id="openai-slash"),
        pytest.param("anthropic/claude-3-5-haiku-latest", "anthropic:claude-3-5-haiku-latest", id="anthropic-slash"),
        pytest.param("google/gemini-2.0-flash", "google:gemini-2.0-flash", id="google-slash"),
        pytest.param("gemini/gemini-2.0-flash", "google:gemini-2.0-flash", id="gemini-slash"),
        pytest.param("openrouter/openai/gpt-4o-mini", "openrouter:openai/gpt-4o-mini", id="openrouter-slash"),
        pytest.param("gpt-4o-mini", "openai:gpt-4o-mini", id="openai-short"),
        pytest.param("claude-3-5-haiku-latest", "anthropic:claude-3-5-haiku-latest", id="anthropic-short"),
        pytest.param("gemini-2.0-flash", "google:gemini-2.0-flash", id="google-short"),
    ],
)
def test_to_pydantic_ai_model_id_maps_common_dspy_slash_ids(model_id: str, expected: str) -> None:
    assert _to_pydantic_ai_model_id(model_id) == expected
