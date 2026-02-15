"""Unit tests for the configuration module."""
import pytest
from unittest.mock import patch
import os


class TestAppConfig:
    """Tests for AppConfig class."""
    
    @pytest.mark.unit
    def test_config_loads_from_environment(self, mock_env_vars):
        """Test that config loads values from environment variables."""
        # Import after setting env vars
        from core.config import AppConfig
        
        config = AppConfig()
        
        assert config.openai_api_key == "test-api-key"
        assert config.openai_api_base == "https://api.openai.com/v1"
        assert config.llama_api_key == "test-llama-key"
    
    @pytest.mark.unit
    def test_config_has_default_values(self, mock_env_vars):
        """Test that config has sensible defaults."""
        from core.config import AppConfig
        
        config = AppConfig()
        
        assert config.chat_model == "gpt-4o-mini"
        assert config.groundedness_threshold == 0.7
        assert config.precision_threshold == 0.7
        assert config.max_refinement_iterations == 3
    
    @pytest.mark.unit
    def test_config_validation_fails_without_api_key(self, monkeypatch):
        """Test that validation fails when required keys are missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        
        from core.config import AppConfig
        
        config = AppConfig()
        
        with pytest.raises(ValueError, match="Configuration errors"):
            config.validate()
    
    @pytest.mark.unit
    def test_config_paths_are_relative_to_project(self, mock_env_vars):
        """Test that paths are relative to project root."""
        from core.config import AppConfig, PROJECT_ROOT
        
        config = AppConfig()
        
        assert PROJECT_ROOT in config.data_dir.parents or config.data_dir == PROJECT_ROOT / "data"
        assert PROJECT_ROOT in config.vector_db_dir.parents or config.vector_db_dir == PROJECT_ROOT / "research_db"


class TestLoggingConfig:
    """Tests for logging configuration."""
    
    @pytest.mark.unit
    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a configured logger."""
        from core.logging_config import setup_logging
        import logging
        
        logger = setup_logging(level=logging.DEBUG)
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    @pytest.mark.unit
    def test_get_logger_returns_named_logger(self):
        """Test that get_logger returns a logger with the correct name."""
        from core.logging_config import get_logger
        
        logger = get_logger("test_module")
        
        assert logger.name == "test_module"
