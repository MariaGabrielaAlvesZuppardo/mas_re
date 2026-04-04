AVAILABLE_MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "context_window": 128_000,
        "supports_json_mode": True,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "context_window": 128_000,
        "supports_json_mode": True,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "context_window": 200_000,
        "supports_json_mode": True,
    },
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "context_window": 200_000,
        "supports_json_mode": True,
    },
}

DEFAULT_MODEL = "gpt-4o"
