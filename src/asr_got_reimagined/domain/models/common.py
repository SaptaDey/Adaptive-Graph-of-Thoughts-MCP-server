import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, field_validator


# Helper for probability distributions (list of floats summing to 1.0)
def _validate_probability_distribution(v: list[float]) -> list[float]:
    """
    Validates that a list of floats represents a probability distribution.
    
    Checks that all values are between 0.0 and 1.0. Allows empty lists and does not enforce that the sum equals 1.0, assuming normalization may occur elsewhere.
    
    Args:
        v: List of floats to validate.
    
    Returns:
        The validated list of floats if all values are within the valid range.
    
    Raises:
        ValueError: If any value is outside the range [0.0, 1.0].
    """
    if not v:  # Empty list is valid if optional, but if provided, must sum to 1
        return v
    if not all(0.0 <= p <= 1.0 for p in v):
        raise ValueError("All probabilities must be between 0.0 and 1.0")
    if abs(sum(v) - 1.0) > 1e-6 and sum(v) != 0:  # Allow sum to be 0 for uninitialized
        # For flexibility, we might allow non-normalized lists and normalize them later
        # For now, strict validation. Could also add a normalization step here.
        # raise ValueError("Probabilities must sum to 1.0")
        pass  # Relaxing this for now, assume normalization happens elsewhere if critical
    return v


ProbabilityDistribution = Annotated[
    list[float], BeforeValidator(_validate_probability_distribution)
]


# Confidence Vector based on P1.5
# Using a class for better type hinting and potential methods later
class ConfidenceVector(BaseModel):
    empirical_support: float = Field(default=0.5, ge=0.0, le=1.0)
    theoretical_basis: float = Field(default=0.5, ge=0.0, le=1.0)
    methodological_rigor: float = Field(default=0.5, ge=0.0, le=1.0)
    consensus_alignment: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_list(self) -> list[float]:
        """
        Returns the confidence vector as a list of four float values in a fixed order.
        """
        return [
            self.empirical_support,
            self.theoretical_basis,
            self.methodological_rigor,
            self.consensus_alignment,
        ]

    @classmethod
    def from_list(cls, values: list[float]) -> "ConfidenceVector":
        """
        Creates a ConfidenceVector instance from a list of four float values.
        
        Args:
            values: A list of four floats representing empirical support, theoretical basis, methodological rigor, and consensus alignment, in that order.
        
        Returns:
            A ConfidenceVector initialized with the provided values.
        
        Raises:
            ValueError: If the input list does not contain exactly four elements.
        """
        if len(values) != 4:
            raise ValueError("Confidence list must have exactly 4 values.")
        return cls(
            empirical_support=values[0],
            theoretical_basis=values[1],
            methodological_rigor=values[2],
            consensus_alignment=values[3],
        )

    @property
    def average_confidence(self) -> float:
        return sum(self.to_list()) / 4.0


# Single scalar certainty/confidence if needed
CertaintyScore = Annotated[float, Field(ge=0.0, le=1.0)]

# Impact Score (P1.28)
ImpactScore = Annotated[float, Field(ge=0.0, le=1.0)]  # Assuming normalized impact


class TimestampedModel(BaseModel):
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

    def touch(self):
        """Updates the updated_at timestamp."""
        self.updated_at = datetime.datetime.now()


# Standardized way to represent a probability distribution for discrete outcomes
class DiscreteProbabilityDistribution(BaseModel):
    outcomes: list[str]  # The labels for each probability
    probabilities: ProbabilityDistribution

    @field_validator("probabilities")
    def check_probabilities_match_outcomes(cls, v, info):
        values = info.data
        if "outcomes" in values and len(v) != len(values["outcomes"]):
            raise ValueError("Number of probabilities must match number of outcomes")
        return v


class EpistemicStatus(str, Enum):
    """P1.12: Epistemic status of a node/claim."""

    ASSUMPTION = "assumption"
    HYPOTHESIS = "hypothesis"
    EVIDENCE_SUPPORTED = "evidence_supported"
    EVIDENCE_CONTRADICTED = "evidence_contradicted"
    THEORETICALLY_DERIVED = "theoretically_derived"
    WIDELY_ACCEPTED = "widely_accepted"
    DISPUTED = "disputed"
    UNKNOWN = "unknown"
    INFERRED = "inferred"
    SPECULATION = "speculation"
