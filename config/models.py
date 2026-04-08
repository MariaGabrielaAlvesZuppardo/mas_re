AVAILABLE_MODELS = {
    "claude-opus-4-6": {
        "provider": "anthropic",
        "category": "LLM",
        "context_window": 200_000,
        "supports_json_mode": True,
        "description": "Maior capacidade — elicitação e priorização complexas",
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "category": "LLM",
        "context_window": 200_000,
        "supports_json_mode": True,
        "description": "Equilíbrio desempenho/custo — padrão do pipeline",
    },
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "category": "SLM",
        "context_window": 200_000,
        "supports_json_mode": True,
        "description": "Modelo compacto e rápido da Anthropic",
    },

    "Qwen/Qwen2.5-0.5B-Instruct": {
        "provider": "huggingface",
        "category": "SLM",
        "context_window": 32_768,
        "supports_json_mode": False,
        "description": "Qwen 2.5 500M — SLM via HuggingFace para comparação com LLMs",
    },

    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "category": "LLM",
        "context_window": 128_000,
        "supports_json_mode": True,
        "description": "Llama 3.3 70B via Groq — gratuito para desenvolvimento",
    },

}

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Agrupamentos para experimentos comparativos
LLM_MODELS = [k for k, v in AVAILABLE_MODELS.items() if v["category"] == "LLM"]
SLM_MODELS = [k for k, v in AVAILABLE_MODELS.items() if v["category"] == "SLM"]
