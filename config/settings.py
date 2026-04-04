from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    default_model: str = "gpt-4o"
    classifier_model: str = "gpt-4o"
    prioritizer_model: str = "gpt-4o"

    max_retries: int = 3
    temperature: float = 0.0

    class Config:
        env_file = ".env"


settings = Settings()
