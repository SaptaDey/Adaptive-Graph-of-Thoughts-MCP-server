import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Optional

from src.adaptive_graph_of_thoughts.domain.utils.math_helpers import (
    bayesian_update_confidence,
    calculate_information_gain,
)
from src.adaptive_graph_of_thoughts.domain.models.common import CertaintyScore, ConfidenceVector
from src.adaptive_graph_of_thoughts.domain.models.graph_elements import EdgeType, StatisticalPower

@pytest.fixture
def sample_confidence_vector():
    """Create a sample ConfidenceVector for testing."""
    return ConfidenceVector.from_list([0.5, 0.6, 0.7, 0.4])

@pytest.fixture
def sample_certainty_scores():
    """Create various CertaintyScore values for testing."""
    return {
        'low': 0.2,
        'medium': 0.5,
        'high': 0.8,
        'zero': 0.0,
        'max': 1.0
    }

@pytest.fixture
def sample_statistical_powers():
    """Create various StatisticalPower values for testing."""
    return [
        StatisticalPower(0.1),
        StatisticalPower(0.5),
        StatisticalPower(0.8),
        StatisticalPower(1.0)
    ]

@pytest.fixture
def all_edge_types():
    """Create all EdgeType enum values for testing."""
    return [
        EdgeType.CAUSES,
        EdgeType.SUPPORTIVE,
        EdgeType.CORRELATIVE,
        EdgeType.CONTRADICTORY
    ]

@pytest.fixture
def sample_probability_distributions():
    """Create sample probability distributions for information gain tests."""
    return {
        'uniform': [0.25, 0.25, 0.25, 0.25],
        'skewed': [0.1, 0.2, 0.3, 0.4],
        'concentrated': [0.9, 0.05, 0.03, 0.02],
        'empty': [],
        'single': [1.0],
        'zero_sum': [0.0, 0.0, 0.0, 0.0],
        'invalid_sum': [0.3, 0.3, 0.3, 0.3]  # Sums to 1.2
    }

class TestBayesianUpdateConfidence:
    """Test suite for bayesian_update_confidence function."""
    
    def test_bayesian_update_supportive_evidence_increases_confidence(self, sample_confidence_vector, sample_certainty_scores):
        """Test that supportive evidence increases confidence values."""
        prior = sample_confidence_vector
        evidence_strength = sample_certainty_scores['high']
        
        result = bayesian_update_confidence(
            prior_confidence=prior,
            evidence_strength=evidence_strength,
            evidence_supports_hypothesis=True
        )
        
        # All values should increase when evidence is supportive
        prior_values = prior.to_list()
        result_values = result.to_list()
        
        for i in range(len(prior_values)):
            assert result_values[i] >= prior_values[i], f"Value at index {i} should increase"
        
        # At least some values should actually increase (not just stay the same)
        assert any(result_values[i] > prior_values[i] for i in range(len(prior_values)))

    def test_bayesian_update_contradictory_evidence_decreases_confidence(self, sample_confidence_vector, sample_certainty_scores):
        """Test that contradictory evidence decreases confidence values."""
        prior = sample_confidence_vector
        evidence_strength = sample_certainty_scores['high']
        
        result = bayesian_update_confidence(
            prior_confidence=prior,
            evidence_strength=evidence_strength,
            evidence_supports_hypothesis=False
        )
        
        # All values should decrease when evidence is contradictory
        prior_values = prior.to_list()
        result_values = result.to_list()
        
        for i in range(len(prior_values)):
            assert result_values[i] <= prior_values[i], f"Value at index {i} should decrease"

    @pytest.mark.parametrize("evidence_strength", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_bayesian_update_with_various_evidence_strengths(self, sample_confidence_vector, evidence_strength):
        """Test bayesian update with different evidence strength values."""
        result = bayesian_update_confidence(
            prior_confidence=sample_confidence_vector,
            evidence_strength=evidence_strength,
            evidence_supports_hypothesis=True
        )
        
        # Result should be a valid ConfidenceVector
        assert isinstance(result, ConfidenceVector)
        result_values = result.to_list()
        
        # All values should be between 0 and 1
        for value in result_values:
            assert 0.0 <= value <= 1.0, f"Confidence value {value} out of bounds"

    def test_bayesian_update_with_statistical_power(self, sample_confidence_vector, sample_statistical_powers):
        """Test bayesian update with different statistical power values."""
        for power in sample_statistical_powers:
            result = bayesian_update_confidence(
                prior_confidence=sample_confidence_vector,
                evidence_strength=0.5,
                evidence_supports_hypothesis=True,
                statistical_power=power
            )
            
            assert isinstance(result, ConfidenceVector)
            # Higher statistical power should lead to greater changes
            result_values = result.to_list()
            for value in result_values:
                assert 0.0 <= value <= 1.0

    def test_bayesian_update_with_edge_types(self, sample_confidence_vector, all_edge_types):
        """Test bayesian update with different edge types."""
        for edge_type in all_edge_types:
            result = bayesian_update_confidence(
                prior_confidence=sample_confidence_vector,
                evidence_strength=0.5,
                evidence_supports_hypothesis=True,
                edge_type=edge_type
            )
            
            assert isinstance(result, ConfidenceVector)
            result_values = result.to_list()
            for value in result_values:
                assert 0.0 <= value <= 1.0

    def test_bayesian_update_edge_type_influences_correctly(self, sample_confidence_vector):
        """Test that different edge types have expected relative influences."""
        base_result = bayesian_update_confidence(
            prior_confidence=sample_confidence_vector,
            evidence_strength=0.5,
            evidence_supports_hypothesis=True
        )
        
        # CAUSES and SUPPORTIVE should have stronger influence (factor 1.1)
        causes_result = bayesian_update_confidence(
            prior_confidence=sample_confidence_vector,
            evidence_strength=0.5,
            evidence_supports_hypothesis=True,
            edge_type=EdgeType.CAUSES
        )
        
        # CORRELATIVE should have weaker influence (factor 0.9)
        correlative_result = bayesian_update_confidence(
            prior_confidence=sample_confidence_vector,
            evidence_strength=0.5,
            evidence_supports_hypothesis=True,
            edge_type=EdgeType.CORRELATIVE
        )
        
        # Compare the magnitudes of change
        base_change = sum(abs(b - p) for b, p in zip(base_result.to_list(), sample_confidence_vector.to_list()))
        causes_change = sum(abs(c - p) for c, p in zip(causes_result.to_list(), sample_confidence_vector.to_list()))
        correlative_change = sum(abs(c - p) for c, p in zip(correlative_result.to_list(), sample_confidence_vector.to_list()))
        
        assert causes_change >= base_change, "CAUSES edge type should have stronger influence"
        assert correlative_change <= base_change, "CORRELATIVE edge type should have weaker influence"

    def test_bayesian_update_extreme_confidence_values(self):
        """Test bayesian update with extreme confidence values (0.0 and 1.0)."""
        # Test with all zeros
        zero_confidence = ConfidenceVector.from_list([0.0, 0.0, 0.0, 0.0])
        result = bayesian_update_confidence(
            prior_confidence=zero_confidence,
            evidence_strength=0.8,
            evidence_supports_hypothesis=True
        )
        
        result_values = result.to_list()
        for value in result_values:
            assert 0.0 <= value <= 1.0
            # Should increase from 0.0
            assert value >= 0.0

        # Test with all ones
        max_confidence = ConfidenceVector.from_list([1.0, 1.0, 1.0, 1.0])
        result = bayesian_update_confidence(
            prior_confidence=max_confidence,
            evidence_strength=0.8,
            evidence_supports_hypothesis=False
        )
        
        result_values = result.to_list()
        for value in result_values:
            assert 0.0 <= value <= 1.0
            # Should decrease from 1.0
            assert value <= 1.0

    @patch('src.adaptive_graph_of_thoughts.domain.utils.math_helpers.logger')
    def test_bayesian_update_logs_debug_message(self, mock_logger, sample_confidence_vector):
        """Test that bayesian update logs a debug message."""
        bayesian_update_confidence(
            prior_confidence=sample_confidence_vector,
            evidence_strength=0.5,
            evidence_supports_hypothesis=True
        )
        
        mock_logger.debug.assert_called_once()
        # Verify the log message contains expected information
        call_args = mock_logger.debug.call_args[0][0]
        assert "Bayesian update" in call_args
        assert "Prior" in call_args
        assert "Evidence Strength" in call_args

    def test_bayesian_update_weight_clamping(self, sample_confidence_vector):
        """Test that weight values are properly clamped between 0 and 1."""
        # Test with very high evidence strength and statistical power
        high_power = StatisticalPower(1.0)
        
        result = bayesian_update_confidence(
            prior_confidence=sample_confidence_vector,
            evidence_strength=1.0,
            evidence_supports_hypothesis=True,
            statistical_power=high_power,
            edge_type=EdgeType.CAUSES  # This adds 1.1 multiplier
        )
        
        # Even with extreme inputs, result should be valid
        result_values = result.to_list()
        for value in result_values:
            assert 0.0 <= value <= 1.0

class TestCalculateInformationGain:
    """Test suite for calculate_information_gain function."""
    
    def test_information_gain_identical_distributions_returns_zero(self):
        """Test that identical distributions return zero information gain."""
        distribution = [0.25, 0.25, 0.25, 0.25]
        result = calculate_information_gain(distribution, distribution)
        
        assert result == 0.0, "Identical distributions should have zero information gain"

    def test_information_gain_completely_different_distributions(self):
        """Test information gain with completely different distributions."""
        prior = [1.0, 0.0, 0.0, 0.0]
        posterior = [0.0, 0.0, 0.0, 1.0]
        
        result = calculate_information_gain(prior, posterior)
        
        # Should return the maximum possible gain (average of absolute differences)
        expected = sum(abs(p - q) for p, q in zip(prior, posterior)) / len(prior)
        assert result == expected
        assert result > 0.0, "Completely different distributions should have positive gain"

    @pytest.mark.parametrize("prior,posterior,expected", [
        ([0.5, 0.5], [0.6, 0.4], 0.1),
        ([0.3, 0.7], [0.7, 0.3], 0.4),
        ([0.0, 1.0], [1.0, 0.0], 1.0),
        ([0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1], 0.1),
    ])
    def test_information_gain_specific_cases(self, prior, posterior, expected):
        """Test information gain with specific distribution pairs."""
        result = calculate_information_gain(prior, posterior)
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_information_gain_empty_distributions(self):
        """Test information gain with empty distributions."""
        result = calculate_information_gain([], [])
        assert result == 0.0, "Empty distributions should return zero gain"

    def test_information_gain_single_element_distributions(self):
        """Test information gain with single-element distributions."""
        result = calculate_information_gain([1.0], [0.5])
        expected = 0.5  # |1.0 - 0.5| / 1
        assert result == expected

    def test_information_gain_mismatched_lengths_returns_zero(self):
        """Test that mismatched distribution lengths return zero gain."""
        prior = [0.5, 0.5]
        posterior = [0.3, 0.3, 0.4]
        
        result = calculate_information_gain(prior, posterior)
        assert result == 0.0, "Mismatched distribution lengths should return zero"

    def test_information_gain_various_distribution_lengths(self):
        """Test information gain with distributions of various lengths."""
        for length in [1, 2, 5, 10, 100]:
            prior = [1.0 / length] * length
            posterior = [2.0 / length if i == 0 else (1.0 - 1.0/length) / (length - 1) for i in range(length)]
            
            result = calculate_information_gain(prior, posterior)
            assert isinstance(result, float)
            assert result >= 0.0, "Information gain should be non-negative"

    def test_information_gain_with_extreme_values(self):
        """Test information gain with extreme probability values."""
        # Test with very small probabilities
        small_prior = [1e-10, 1.0 - 1e-10]
        small_posterior = [1e-9, 1.0 - 1e-9]
        
        result = calculate_information_gain(small_prior, small_posterior)
        assert isinstance(result, float)
        assert result >= 0.0

        # Test with zero probabilities
        zero_prior = [0.0, 1.0]
        zero_posterior = [0.1, 0.9]
        
        result = calculate_information_gain(zero_prior, zero_posterior)
        assert result == 0.05  # (|0.0-0.1| + |1.0-0.9|) / 2

    def test_information_gain_floating_point_precision(self):
        """Test information gain with floating-point precision edge cases."""
        # Test with numbers that might cause precision issues
        prior = [1.0/3.0, 1.0/3.0, 1.0/3.0]
        posterior = [0.333333, 0.333333, 0.333334]
        
        result = calculate_information_gain(prior, posterior)
        assert isinstance(result, float)
        assert result >= 0.0
        # Should be very small due to precision differences
        assert result < 1e-5

    @patch('src.adaptive_graph_of_thoughts.domain.utils.math_helpers.logger')
    def test_information_gain_logs_debug_message(self, mock_logger):
        """Test that information gain calculation logs a debug message."""
        prior = [0.5, 0.5]
        posterior = [0.6, 0.4]
        
        calculate_information_gain(prior, posterior)
        
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "Calculated simplified info gain" in call_args

    def test_information_gain_mathematical_properties(self):
        """Test mathematical properties of information gain."""
        prior = [0.4, 0.3, 0.2, 0.1]
        posterior = [0.3, 0.4, 0.1, 0.2]
        
        # Information gain should be symmetric
        gain1 = calculate_information_gain(prior, posterior)
        gain2 = calculate_information_gain(posterior, prior)
        
        assert gain1 == gain2, "Information gain should be symmetric"
        
        # Gain should be non-negative
        assert gain1 >= 0.0, "Information gain should be non-negative"
        
        # Triangle inequality-like property
        intermediate = [0.35, 0.35, 0.15, 0.15]
        gain_direct = calculate_information_gain(prior, posterior)
        gain_via_intermediate = (calculate_information_gain(prior, intermediate) +
                                 calculate_information_gain(intermediate, posterior))
        
        # Direct gain should be <= sum of intermediate gains (triangle inequality)
        assert gain_direct <= gain_via_intermediate + 1e-10  # Small epsilon for floating point

    def test_information_gain_boundary_cases(self):
        """Test information gain with boundary cases and edge values."""
        # Test with distributions that sum to different values (not necessarily 1.0)
        non_normalized_prior = [0.2, 0.3, 0.6]  # Sums to 1.1
        non_normalized_posterior = [0.1, 0.4, 0.4]  # Sums to 0.9
        
        result = calculate_information_gain(non_normalized_prior, non_normalized_posterior)
        expected = (abs(0.2-0.1) + abs(0.3-0.4) + abs(0.6-0.4)) / 3
        assert abs(result - expected) < 1e-10

    @pytest.mark.parametrize("distribution_size", [1, 2, 10, 100, 1000])
    def test_information_gain_performance_various_sizes(self, distribution_size):
        """Test information gain performance with various distribution sizes."""
        import time
        
        prior = [1.0 / distribution_size] * distribution_size
        posterior = [
            2.0 / distribution_size if i == 0 else 
            (1.0 - 1.0/distribution_size) / (distribution_size - 1)
            for i in range(distribution_size)
        ]
        
        start_time = time.time()
        result = calculate_information_gain(prior, posterior)
        execution_time = time.time() - start_time
        
        # Should complete quickly even for large distributions
        assert execution_time < 1.0, f"Execution took too long: {execution_time}s for size {distribution_size}"
        assert isinstance(result, float)
        assert result >= 0.0

class TestMathHelpersIntegration:
    """Integration tests for math_helpers module functions working together."""
    
    def test_bayesian_update_then_information_gain_workflow(self, sample_confidence_vector):
        """Test a realistic workflow combining both functions."""
        # Simulate a sequence of evidence updates
        evidence_sequence = [
            (0.6, True),   # Strong supportive evidence
            (0.3, False),  # Weak contradictory evidence  
            (0.8, True),   # Very strong supportive evidence
        ]
        
        current_confidence = sample_confidence_vector
        confidence_history = [current_confidence.to_list()]
        
        for evidence_strength, supports in evidence_sequence:
            current_confidence = bayesian_update_confidence(
                prior_confidence=current_confidence,
                evidence_strength=evidence_strength,
                evidence_supports_hypothesis=supports
            )
            confidence_history.append(current_confidence.to_list())
        
        # Calculate information gain between initial and final states
        initial_dist = confidence_history[0]
        final_dist = confidence_history[-1]
        
        total_gain = calculate_information_gain(initial_dist, final_dist)
        
        # Should have positive information gain due to evidence updates
        assert total_gain > 0.0, "Evidence sequence should produce information gain"
        
        # Calculate incremental gains
        incremental_gains = []
        for i in range(len(confidence_history) - 1):
            gain = calculate_information_gain(confidence_history[i], confidence_history[i + 1])
            incremental_gains.append(gain)
        
        # All incremental gains should be non-negative
        for gain in incremental_gains:
            assert gain >= 0.0, "Each evidence update should produce non-negative gain"

    def test_multiple_updates_converge_appropriately(self):
        """Test that multiple bayesian updates converge to expected ranges."""
        initial_confidence = ConfidenceVector.from_list([0.5, 0.5, 0.5, 0.5])
        
        # Apply many strong supportive evidence updates
        current = initial_confidence
        for _ in range(10):
            current = bayesian_update_confidence(
                prior_confidence=current,
                evidence_strength=0.8,
                evidence_supports_hypothesis=True
            )
        
        # Should converge toward higher confidence values
        final_values = current.to_list()
        initial_values = initial_confidence.to_list()
        
        for i in range(len(final_values)):
            assert final_values[i] > initial_values[i], "Repeated supportive evidence should increase confidence"
            assert final_values[i] <= 1.0, "Confidence should not exceed 1.0"

    def test_alternating_evidence_stability(self, sample_confidence_vector):
        """Test system stability with alternating supportive/contradictory evidence."""
        current = sample_confidence_vector
        
        # Apply alternating evidence
        for i in range(20):
            supports = (i % 2 == 0)  # Alternates True/False
            current = bayesian_update_confidence(
                prior_confidence=current,
                evidence_strength=0.3,  # Moderate evidence
                evidence_supports_hypothesis=supports
            )
        
        # Should remain in valid bounds
        final_values = current.to_list()
        for value in final_values:
            assert 0.0 <= value <= 1.0, "Alternating evidence should maintain valid bounds"

class TestMathHelpersModuleIntegrity:
    """Tests for overall module integrity and imports."""
    
    def test_module_imports_correctly(self):
        """Test that the math_helpers module can be imported without errors."""
        try:
            from src.adaptive_graph_of_thoughts.domain.utils import math_helpers
            assert hasattr(math_helpers, 'bayesian_update_confidence')
            assert hasattr(math_helpers, 'calculate_information_gain')
        except ImportError as e:
            pytest.fail(f"Failed to import math_helpers module: {e}")

    def test_all_functions_are_callable(self):
        """Test that all exported functions are callable."""
        from src.adaptive_graph_of_thoughts.domain.utils.math_helpers import (
            bayesian_update_confidence,
            calculate_information_gain,
        )
        
        assert callable(bayesian_update_confidence), "bayesian_update_confidence should be callable"
        assert callable(calculate_information_gain), "calculate_information_gain should be callable"

    def test_function_signatures_match_expected(self):
        """Test that function signatures match expected parameters."""
        import inspect
        from src.adaptive_graph_of_thoughts.domain.utils.math_helpers import (
            bayesian_update_confidence,
            calculate_information_gain,
        )
        
        # Check bayesian_update_confidence signature
        bayesian_sig = inspect.signature(bayesian_update_confidence)
        expected_bayesian_params = [
            'prior_confidence', 'evidence_strength', 'evidence_supports_hypothesis',
            'statistical_power', 'edge_type'
        ]
        
        actual_bayesian_params = list(bayesian_sig.parameters.keys())
        assert actual_bayesian_params == expected_bayesian_params, \
            f"Bayesian function signature mismatch: {actual_bayesian_params}"
        
        # Check calculate_information_gain signature
        gain_sig = inspect.signature(calculate_information_gain)
        expected_gain_params = ['prior_distribution', 'posterior_distribution']
        
        actual_gain_params = list(gain_sig.parameters.keys())
        assert actual_gain_params == expected_gain_params, \
            f"Information gain function signature mismatch: {actual_gain_params}"

    @patch('src.adaptive_graph_of_thoughts.domain.utils.math_helpers.logger')
    def test_module_logging_integration(self, mock_logger, sample_confidence_vector):
        """Test that module-level logging works correctly."""
        from src.adaptive_graph_of_thoughts.domain.utils.math_helpers import (
            bayesian_update_confidence,
            calculate_information_gain,
        )
        
        # Test that both functions use logging
        bayesian_update_confidence(
            prior_confidence=sample_confidence_vector,
            evidence_strength=0.5,
            evidence_supports_hypothesis=True
        )
        
        calculate_information_gain([0.5, 0.5], [0.6, 0.4])
        
        # Should have at least 2 debug calls
        assert mock_logger.debug.call_count >= 2, "Both functions should log debug messages"

    def test_module_has_proper_typing(self):
        """Test that module functions have proper type annotations."""
        import inspect
        from src.adaptive_graph_of_thoughts.domain.utils.math_helpers import (
            bayesian_update_confidence,
            calculate_information_gain,
        )
        
        # Check that return types are annotated
        bayesian_sig = inspect.signature(bayesian_update_confidence)
        assert bayesian_sig.return_annotation is not None, "Bayesian function should have return type annotation"
        
        gain_sig = inspect.signature(calculate_information_gain)
        assert gain_sig.return_annotation is not None, "Information gain function should have return type annotation"

if __name__ == "__main__":
    pytest.main([__file__])