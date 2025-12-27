"""Configuration management for Brainstormer."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM API Keys
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["openai", "local"] = "openai"
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # Web Search
    tavily_api_key: str | None = None

    # Database paths
    sqlite_db_path: Path = Field(default_factory=lambda: Path("./brainstormer.db"))
    chromadb_path: Path = Field(default_factory=lambda: Path("./chromadb"))

    # Skills directory
    skills_dir: Path = Field(default_factory=lambda: Path("./skills"))

    # Default LLM settings
    default_llm_provider: Literal["anthropic", "openai", "openrouter"] = "anthropic"
    default_llm_model: str = "claude-sonnet-4-5-20250929"

    # Logging
    log_level: str = "INFO"

    def get_model_string(self) -> str:
        """Get the model string for langchain init_chat_model."""
        return f"{self.default_llm_provider}:{self.default_llm_model}"

    def validate_api_keys(self) -> list[str]:
        """Validate that required API keys are present."""
        errors = []
        if self.default_llm_provider == "anthropic" and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required when using Anthropic provider")
        if self.default_llm_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required when using OpenAI provider")
        if self.default_llm_provider == "openrouter" and not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is required when using OpenRouter provider")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for OpenAI embeddings")
        return errors


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from environment and optional .env file."""
    if env_file and env_file.exists():
        # pydantic-settings v2 uses _env_file in constructor
        return Settings(_env_file=env_file)  # type: ignore[call-arg]
    return Settings()
