"""Constants for the LLM Sandbox integration."""

from typing import Final

DOMAIN: Final = "llm_sandbox"
DEFAULT_NAME: Final = "Assistant Sandbox"
DEFAULT_ASSISTANT: Final = "conversation"

CONF_ASSISTANT: Final = "assistant"
CONF_NAME: Final = "name"
CONF_EXECUTION_TIMEOUT: Final = "execution_timeout_seconds"
CONF_HELPER_CALL_BUDGET: Final = "helper_call_budget"

DEFAULT_EXECUTION_TIMEOUT_SECONDS: Final = 12
DEFAULT_HELPER_CALL_BUDGET: Final = 32

MIN_EXECUTION_TIMEOUT_SECONDS: Final = 3
MAX_EXECUTION_TIMEOUT_SECONDS: Final = 30
MIN_HELPER_CALL_BUDGET: Final = 1
MAX_HELPER_CALL_BUDGET: Final = 100

# LLM tool name. Mirrors the Home Assistant convention for code execution tools.
TOOL_EXECUTE_HOME_CODE: Final = "execute_home_code"
