from enum import Enum
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GEMINI = "google_genai"
    GOOGLE_VERTEX = "google_vertex"


class BaseModelSettings(BaseSettings, case_sensitive=False):
    model_config = SettingsConfigDict(env_prefix="model_", env_file=".env", env_file_encoding="utf-8", extra="allow")

    provider: str = Field("openai", max_length=200, min_length=5)
    model_name: str = Field("gpt-3.5-turbo", alias="model_name", max_length=200, min_length=5)
    temperature: float = Field(0.2)
    max_tokens: int = Field(100)

    @field_validator('provider', mode='after')
    @classmethod
    def validate_provider(cls, value: str) -> str:
        is_valid = value in (member.value for member in ModelProvider)
        if not is_valid:
            raise ValueError('Invalid model provider value')
        return value
