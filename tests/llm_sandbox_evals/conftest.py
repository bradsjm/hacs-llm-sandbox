import pytest

from llm_sandbox_evals import cli


@pytest.fixture(autouse=True)
def _disable_logfire_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent eval tests from exporting developer-local telemetry."""
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: False)
