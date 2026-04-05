from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    huggingface_api_key: str = ""

    default_model: str = "claude-sonnet-4-6"
    classifier_model: str = "claude-haiku-4-5-20251001"
    prioritizer_model: str = "claude-opus-4-6"

    max_retries: int = 3
    temperature: float = 0.0

    class Config:
        env_file = ".env"


settings = Settings()
