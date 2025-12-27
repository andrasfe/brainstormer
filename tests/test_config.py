"""Tests for configuration module."""



from brainstormer.config import Settings, load_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.default_llm_provider == "anthropic"
        assert settings.default_llm_model == "claude-sonnet-4-5-20250929"
        assert settings.embedding_provider == "openai"
        assert settings.log_level == "INFO"

    def test_get_model_string(self):
        """Test model string generation."""
        settings = Settings(
            default_llm_provider="anthropic",
            default_llm_model="claude-sonnet-4-5-20250929",
        )

        assert settings.get_model_string() == "anthropic:claude-sonnet-4-5-20250929"

        settings.default_llm_provider = "openai"
        settings.default_llm_model = "gpt-4o"
        assert settings.get_model_string() == "openai:gpt-4o"

    def test_validate_api_keys_anthropic(self):
        """Test API key validation for Anthropic provider."""
        settings = Settings(
            default_llm_provider="anthropic",
            anthropic_api_key=None,
        )
        errors = settings.validate_api_keys()
        assert any("ANTHROPIC_API_KEY" in e for e in errors)

        settings.anthropic_api_key = "valid-key"
        errors = settings.validate_api_keys()
        assert not any("ANTHROPIC_API_KEY" in e for e in errors)

    def test_validate_api_keys_openai(self):
        """Test API key validation for OpenAI provider."""
        settings = Settings(
            default_llm_provider="openai",
            openai_api_key=None,
        )
        errors = settings.validate_api_keys()
        assert any("OPENAI_API_KEY" in e for e in errors)

    def test_validate_api_keys_openrouter(self):
        """Test API key validation for OpenRouter provider."""
        settings = Settings(
            default_llm_provider="openrouter",
            openrouter_api_key=None,
        )
        errors = settings.validate_api_keys()
        assert any("OPENROUTER_API_KEY" in e for e in errors)

        settings.openrouter_api_key = "valid-key"
        errors = settings.validate_api_keys()
        assert not any("OPENROUTER_API_KEY" in e for e in errors)

    def test_load_settings_from_env(self, temp_dir, monkeypatch):
        """Test loading settings from environment variables."""
        # Settings now reads from environment variables via pydantic-settings
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        settings = load_settings()
        assert settings.anthropic_api_key == "test-key-123"
        assert settings.log_level == "DEBUG"
