from enum import Enum
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GEMINI = "google_genai"
    GOOGLE_VERTEX = "google_vertex"

class VectorStoreProvider(str, Enum):
    IN_MEMORY = "in_memory"
    CHROMA = "chroma"
    ELASTIC_SEARCH = "elastic_search"
    MONGODB = "mongodb"
    COSMOSDB = "cosmosdb"
    PG_VECTOR = "pg_vector"
    REDIS = "redis"
    WEAVIATE = "weaviate"

class PDFLoader(str, Enum):
    FIRECRAWL_ANYDOC = "firecrawl-anydoc"
    DOC7 = "doc7"
    DEFAULT = "default"


class LangFuseSettings(BaseSettings, case_sensitive=False):
    model_config = SettingsConfigDict(env_prefix="langfuse_", env_file=".env", env_file_encoding="utf-8", extra="allow")

    base_url: str = Field("http://localhost:3000", max_length=300, min_length=5)
    public_key: str = Field(max_length=100, min_length=2)
    secret_key: str = Field(max_length=100, min_length=2)

class BaseModelSettings(BaseSettings, case_sensitive=False):
    model_config = SettingsConfigDict(env_prefix="model_", env_file=".env", env_file_encoding="utf-8", extra="allow")

    provider: str = Field("openai", max_length=200, min_length=5)
    model_name: str = Field("gpt-3.5-turbo", alias="model_name", max_length=200, min_length=5)
    temperature: float = Field(0.2)
    max_tokens: int = Field(1000)
    verbose: bool = Field(False)

    @field_validator('provider', mode='after')
    @classmethod
    def validate_provider(cls, value: str) -> str:
        is_valid = value in (member.value for member in ModelProvider)
        if not is_valid:
            raise ValueError('Invalid model provider value')
        return value


class BaseToolSettings(BaseSettings, case_sensitive=False):
    model_config = SettingsConfigDict(env_prefix="tool_", env_file=".env", env_file_encoding="utf-8", extra="allow")

    weather_apikey: str = Field(max_length=200, min_length=5)
    weather_url: str = Field(max_length=1000, min_length=5)
    pdf_loader: str = Field("default", max_length=50, min_length=2)

    @field_validator('pdf_loader', mode='after')
    @classmethod
    def validate_pdf_loader(cls, value: str) -> str:
        is_valid = value in (member.value for member in PDFLoader)
        if not is_valid:
            raise ValueError('Invalid custom pdf loader value')
        return value

class DatabaseSettings(BaseSettings, case_sensitive=False):
    model_config = SettingsConfigDict(env_prefix="db_", env_file=".env", env_file_encoding="utf-8", extra="allow")

    url: str = Field(max_length=200, min_length=5)
    raw_url: str = Field(max_length=200, min_length=5)
    username: str = Field(max_length=20, min_length=2, alias="db_user")
    password: str = Field(max_length=20, min_length=2, alias="db_pass")
    database_name: str = Field(max_length=50, min_length=2, alias="db_name")

class StoreSettings(BaseSettings, case_sensitive=False):
    model_config = SettingsConfigDict(env_prefix="store_", env_file=".env", env_file_encoding="utf-8", extra="allow")

    provider: str = Field("pg_vector", max_length=200, min_length=5)
    url: str = Field(max_length=200, min_length=5)
    host: str = Field("localhost", max_length=200, min_length=5)
    port: int = Field(8080)
    username: str = Field(max_length=20, min_length=2, alias="store_user")
    password: str = Field(max_length=20, min_length=2, alias="store_pass")
    store_name: str = Field(max_length=50, min_length=2, alias="store_name")

    @field_validator('provider', mode='after')
    @classmethod
    def validate_provider(cls, value: str) -> str:
        is_valid = value in (member.value for member in VectorStoreProvider)
        if not is_valid:
            raise ValueError('Invalid vector store provider value')
        return value
