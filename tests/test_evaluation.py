"""Unit tests for the evaluation module."""
import pytest
from unittest.mock import patch, MagicMock


class TestScoreParsing:
    """Tests for score parsing functionality."""
    
    @pytest.mark.unit
    def test_parse_score_valid_float(self):
        """Test parsing a valid float score."""
        from core.evaluation import _parse_score
        
        assert _parse_score("0.85") == 0.85
        assert _parse_score("1.0") == 1.0
        assert _parse_score("0.0") == 0.0
    
    @pytest.mark.unit
    def test_parse_score_with_whitespace(self):
        """Test parsing score with surrounding whitespace."""
        from core.evaluation import _parse_score
        
        assert _parse_score("  0.75  ") == 0.75
        assert _parse_score("\n0.9\n") == 0.9
    
    @pytest.mark.unit
    def test_parse_score_with_text(self):
        """Test extracting score from text response."""
        from core.evaluation import _parse_score
        
        assert _parse_score("The score is 0.8") == 0.8
        assert _parse_score("Score: 0.65 based on analysis") == 0.65
    
    @pytest.mark.unit
    def test_parse_score_clamps_to_valid_range(self):
        """Test that scores are clamped to 0.0-1.0 range."""
        from core.evaluation import _parse_score
        
        assert _parse_score("1.5") == 1.0
        assert _parse_score("-0.5") == 0.0
        assert _parse_score("2.0") == 1.0
    
    @pytest.mark.unit
    def test_parse_score_invalid_returns_default(self):
        """Test that invalid input returns default score."""
        from core.evaluation import _parse_score
        
        assert _parse_score("no numbers here") == 0.5
        assert _parse_score("") == 0.5


class TestGroundednessScoring:
    """Tests for groundedness scoring."""
    
    @pytest.mark.unit
    def test_score_groundedness_updates_state(self, sample_agent_state, mock_llm):
        """Test that score_groundedness updates the state correctly."""
        with patch("core.evaluation.llm", mock_llm):
            from core.evaluation import score_groundedness
            
            result = score_groundedness(sample_agent_state)
            
            assert result["groundedness_score"] == 0.85
            assert result["groundedness_loop_count"] == 1
    
    @pytest.mark.unit
    def test_score_groundedness_increments_counter(self, sample_agent_state, mock_llm):
        """Test that loop counter increments on each call."""
        sample_agent_state["groundedness_loop_count"] = 2
        
        with patch("core.evaluation.llm", mock_llm):
            from core.evaluation import score_groundedness
            
            result = score_groundedness(sample_agent_state)
            
            assert result["groundedness_loop_count"] == 3


class TestPrecisionScoring:
    """Tests for precision scoring."""
    
    @pytest.mark.unit
    def test_check_precision_updates_state(self, sample_agent_state, mock_llm):
        """Test that check_precision updates the state correctly."""
        with patch("core.evaluation.llm", mock_llm):
            from core.evaluation import check_precision
            
            result = check_precision(sample_agent_state)
            
            assert result["precision_score"] == 0.85
            assert result["precision_loop_count"] == 1
