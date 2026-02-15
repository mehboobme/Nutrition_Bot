"""Unit tests for the routing module."""
import pytest
from unittest.mock import patch


class TestGroundednessRouting:
    """Tests for groundedness routing decisions."""
    
    @pytest.mark.unit
    def test_routes_to_precision_when_score_above_threshold(self, sample_agent_state):
        """Test routing to precision check when groundedness passes."""
        sample_agent_state["groundedness_score"] = 0.8
        sample_agent_state["groundedness_loop_count"] = 1
        
        from core.routing import should_continue_groundedness
        
        result = should_continue_groundedness(sample_agent_state)
        
        assert result == "check_precision"
    
    @pytest.mark.unit
    def test_routes_to_refine_when_score_below_threshold(self, sample_agent_state):
        """Test routing to refinement when groundedness fails."""
        sample_agent_state["groundedness_score"] = 0.5
        sample_agent_state["groundedness_loop_count"] = 1
        
        from core.routing import should_continue_groundedness
        
        result = should_continue_groundedness(sample_agent_state)
        
        assert result == "refine_response"
    
    @pytest.mark.unit
    def test_routes_to_max_iterations_when_limit_reached(self, sample_agent_state):
        """Test routing to max iterations when loop limit reached."""
        sample_agent_state["groundedness_score"] = 0.5
        sample_agent_state["groundedness_loop_count"] = 3
        
        from core.routing import should_continue_groundedness
        
        result = should_continue_groundedness(sample_agent_state)
        
        assert result == "max_iterations_reached"
    
    @pytest.mark.unit
    def test_exact_threshold_passes(self, sample_agent_state):
        """Test that exact threshold value passes."""
        sample_agent_state["groundedness_score"] = 0.7
        
        from core.routing import should_continue_groundedness
        
        result = should_continue_groundedness(sample_agent_state)
        
        assert result == "check_precision"


class TestPrecisionRouting:
    """Tests for precision routing decisions."""
    
    @pytest.mark.unit
    def test_routes_to_pass_when_score_above_threshold(self, sample_agent_state):
        """Test routing to pass when precision passes."""
        sample_agent_state["precision_score"] = 0.8
        sample_agent_state["precision_loop_count"] = 1
        
        from core.routing import should_continue_precision
        
        result = should_continue_precision(sample_agent_state)
        
        assert result == "pass"
    
    @pytest.mark.unit
    def test_routes_to_refine_query_when_score_below_threshold(self, sample_agent_state):
        """Test routing to query refinement when precision fails."""
        sample_agent_state["precision_score"] = 0.5
        sample_agent_state["precision_loop_count"] = 1
        
        from core.routing import should_continue_precision
        
        result = should_continue_precision(sample_agent_state)
        
        assert result == "refine_query"
    
    @pytest.mark.unit
    def test_routes_to_max_iterations_when_limit_reached(self, sample_agent_state):
        """Test routing to max iterations when loop limit reached."""
        sample_agent_state["precision_score"] = 0.5
        sample_agent_state["precision_loop_count"] = 4
        
        from core.routing import should_continue_precision
        
        result = should_continue_precision(sample_agent_state)
        
        assert result == "max_iterations_reached"


class TestMaxIterationsReached:
    """Tests for max iterations handler."""
    
    @pytest.mark.unit
    def test_sets_fallback_response(self, sample_agent_state):
        """Test that fallback response is set when max iterations reached."""
        from core.routing import max_iterations_reached
        
        result = max_iterations_reached(sample_agent_state)
        
        assert "apologize" in result["response"].lower()
        assert "more" in result["response"].lower() or "context" in result["response"].lower()
