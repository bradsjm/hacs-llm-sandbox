"""Optional Logfire configuration for native pydantic-evals runs."""


def configure_logfire() -> None:
    """Configure token-enabled telemetry without writing to terminal streams."""
    import logfire

    logfire.configure(send_to_logfire="if-token-present", service_name="llm-sandbox-evals", console=False)
    logfire.instrument_pydantic_ai()
