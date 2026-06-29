"""Constants for the LLM Sandbox integration."""

from typing import Final

DOMAIN: Final = "llm_sandbox"
DEFAULT_NAME: Final = "Assistant Sandbox"
DEFAULT_ASSISTANT: Final = "conversation"

CONF_ASSISTANT: Final = "assistant"
CONF_NAME: Final = "name"
CONF_EXECUTION_TIMEOUT: Final = "execution_timeout_seconds"
CONF_HELPER_CALL_BUDGET: Final = "helper_call_budget"
CONF_RESTRICT_TO_ASSIST_EXPOSED: Final = "restrict_to_assist_exposed"
CONF_EXCLUDE_HIDDEN: Final = "exclude_hidden"
CONF_EXCLUDE_CONFIG: Final = "exclude_config"
CONF_EXCLUDE_DIAGNOSTIC: Final = "exclude_diagnostic"
CONF_ACTIONS_ENABLED: Final = "actions_enabled"
CONF_ACTION_DOMAINS: Final = "action_domains"
SECTION_VISIBILITY: Final = "section_visibility"
SECTION_ACTIONS: Final = "section_actions"

DEFAULT_EXECUTION_TIMEOUT_SECONDS: Final = 12
DEFAULT_HELPER_CALL_BUDGET: Final = 32
DEFAULT_RESTRICT_TO_ASSIST_EXPOSED: Final = True
DEFAULT_EXCLUDE_HIDDEN: Final = True
DEFAULT_EXCLUDE_CONFIG: Final = True
DEFAULT_EXCLUDE_DIAGNOSTIC: Final = True
DEFAULT_ACTIONS_ENABLED: Final = False
DEFAULT_ACTION_DOMAINS: Final[tuple[str, ...]] = ()

MIN_EXECUTION_TIMEOUT_SECONDS: Final = 3
MAX_EXECUTION_TIMEOUT_SECONDS: Final = 30
MIN_HELPER_CALL_BUDGET: Final = 1
MAX_HELPER_CALL_BUDGET: Final = 100

# LLM tool name. Mirrors the Home Assistant convention for code execution tools.
TOOL_EXECUTE_HOME_CODE: Final = "execute_home_code"
