from pydantic_settings import BaseSettings
from pydantic import model_validator

class Settings(BaseSettings):
    anthropic_api_key:  str = ""
    groq_api_key:       str = ""
    huggingface_api_key: str = ""

    default_model:    str = "llama-3.3-70b-versatile"
    classifier_model: str = "llama-3.3-70b-versatile"
    prioritizer_model: str = "llama-3.3-70b-versatile"

    max_retries: int = 3
    temperature: float = 0.0

    @model_validator(mode="after")
    def check_api_keys(self) -> "Settings":
        import warnings
        if not self.anthropic_api_key:
            warnings.warn("ANTHROPIC_API_KEY não definida", UserWarning, stacklevel=2)
        if not self.groq_api_key:
            warnings.warn("GROQ_API_KEY não definida", UserWarning, stacklevel=2)
        return self

    class Config:
        env_file = ".env"


settings = Settings()






