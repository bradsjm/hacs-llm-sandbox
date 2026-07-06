"""Optional Logfire configuration for native pydantic-evals runs."""


def configure_logfire() -> None:
    """Configure Logfire only when the eval CLI explicitly asks for it."""
    import logfire

    logfire.configure(send_to_logfire="if-token-present", service_name="llm-sandbox-evals")
