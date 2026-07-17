"""Project-wide pytest configuration."""
import os

# Disable remote telemetry before pytest imports plugins or test modules.
os.environ["LOGFIRE_SEND_TO_LOGFIRE"] = "false"
os.environ.pop("LOGFIRE_TOKEN", None)


pytest_plugins = "pytest_homeassistant_custom_component"
