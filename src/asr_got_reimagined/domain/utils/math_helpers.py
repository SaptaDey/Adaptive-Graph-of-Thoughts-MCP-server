from typing import Optional

from loguru import logger

from src.asr_got_reimagined.domain.models.common import CertaintyScore, ConfidenceVector
from src.asr_got_reimagined.domain.models.graph_elements import (
    EdgeType,
    StatisticalPower,
)


def bayesian_update_confidence(
    prior_confidence: ConfidenceVector,
    evidence_strength: CertaintyScore,  # A single score representing how strong this piece of evidence is
    evidence_supports_hypothesis: bool,  # True if supportive, False if contradictory
    statistical_power: Optional[StatisticalPower] = None,  # P1.26
    edge_type: Optional[EdgeType] = None,  # P1.10, P1.24, P1.25 for context
) -> ConfidenceVector:
    """
    Performs a simplified Bayesian-inspired update on a confidence vector based on new evidence.
    
    The update adjusts each component of the confidence vector toward 1.0 if the evidence supports the hypothesis, or toward 0.0 if it contradicts, scaled by a weight derived from evidence strength, statistical power, and edge type. Returns a new confidence vector reflecting the updated values.
    """
    # This is a highly simplified placeholder. True Bayesian updates involve likelihoods, priors, etc.
    # and updating probability distributions, not just scores.
    # P1.14 mentions "probability distributions" for confidence components.

    # For now, we'll do a weighted adjustment.
    # Weight factor considers evidence strength and statistical power.
    power_multiplier = (
        statistical_power.value if statistical_power else 0.5
    )  # Default if no power info
    weight = evidence_strength * power_multiplier

    # Edge type influence (P1.14) - very simplified
    edge_type_factor = 1.0
    if edge_type:
        if edge_type in [EdgeType.CAUSES, EdgeType.SUPPORTIVE]:
            edge_type_factor = 1.1
        elif edge_type == EdgeType.CORRELATIVE:
            edge_type_factor = 0.9
        elif edge_type == EdgeType.CONTRADICTORY:
            # This case should be handled by evidence_supports_hypothesis=False
            pass

    weight *= edge_type_factor
    weight = max(0, min(weight, 1.0))  # Clamp weight

    new_confidence_values = prior_confidence.to_list()
    target_value = 1.0 if evidence_supports_hypothesis else 0.0

    # Adjust each component of the confidence vector
    # A more sophisticated approach would update each dimension based on how the evidence relates to it.
    # E.g., empirical evidence primarily boosts empirical_support.
    # This simple version boosts/reduces all components slightly.
    for i in range(len(new_confidence_values)):
        current_val = new_confidence_values[i]
        adjustment = weight * (target_value - current_val)
        new_confidence_values[i] = max(0.0, min(1.0, current_val + adjustment))

    logger.debug(
        f"Bayesian update: Prior {prior_confidence.to_list()}, Evidence Strength {evidence_strength}, "
        f"Supports: {evidence_supports_hypothesis}, Power: {power_multiplier}, Edge: {edge_type} "
        f"-> New {new_confidence_values}"
    )
    return ConfidenceVector.from_list(new_confidence_values)


def calculate_information_gain(
    prior_distribution: list[float], posterior_distribution: list[float]
) -> float:
    """
    Calculates a simplified information gain between prior and posterior probability distributions.
    
    Returns the average absolute difference between corresponding elements of the two distributions. If the distributions differ in length, returns 0.0.
    """
    # Simplified: sum of absolute changes in probability for now
    if len(prior_distribution) != len(posterior_distribution):
        return 0.0  # Or raise error

    gain = sum(
        abs(p - q) for p, q in zip(prior_distribution, posterior_distribution)
    ) / len(prior_distribution)
    logger.debug(f"Calculated simplified info gain: {gain}")
    return gain
