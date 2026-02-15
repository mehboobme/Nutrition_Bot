"""Unit tests for the guardrails module."""
import pytest
from unittest.mock import patch, MagicMock


class TestLlamaGuard:
    """Tests for Llama Guard safety filtering."""
    
    @pytest.mark.unit
    def test_filter_returns_safe_for_normal_input(self):
        """Test that normal input is classified as safe."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="SAFE"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("agents.guard._get_llama_guard_client", return_value=mock_client):
            from agents.guard import filter_input_with_llama_guard
            
            result = filter_input_with_llama_guard("What is vitamin D?")
            
            assert result == "SAFE"
    
    @pytest.mark.unit
    def test_filter_returns_safe_when_client_not_available(self):
        """Test graceful degradation when client is not configured."""
        with patch("agents.guard._get_llama_guard_client", return_value=None):
            from agents.guard import filter_input_with_llama_guard
            
            result = filter_input_with_llama_guard("Test input")
            
            assert result == "SAFE"
    
    @pytest.mark.unit
    def test_filter_returns_none_on_error(self):
        """Test that errors return None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch("agents.guard._get_llama_guard_client", return_value=mock_client):
            from agents.guard import filter_input_with_llama_guard
            
            result = filter_input_with_llama_guard("Test input")
            
            assert result is None
    
    @pytest.mark.unit
    def test_client_creation_without_api_key(self, monkeypatch):
        """Test that client returns None without API key."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        
        from agents.guard import _get_llama_guard_client
        
        # Force reimport to pick up env change
        import importlib
        import agents.guard
        importlib.reload(agents.guard)
        
        client = agents.guard._get_llama_guard_client()
        
        assert client is None
