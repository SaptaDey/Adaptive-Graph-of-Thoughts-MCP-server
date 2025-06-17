"""
Scoring models for ASR-GoT system.
"""

from pydantic import BaseModel, Field


class ScoreResult(BaseModel):
    """
    Model for scoring results in the ASR-GoT system.

    This model is used to represent scoring results from the GoTProcessor,
    including confidence scores, metrics, and other evaluation data.
    """

    score: float = Field(default=0.0, description="Overall score value between 0 and 1")
    confidence_vector: float = Field(
        default=0.0, description="Simplified confidence value"
    )
    metrics: str = Field(default="", description="Additional metrics as string")
    details: str = Field(default="", description="Additional details as string")
    category_scores: str = Field(
        default="", description="Scores broken down by category"
    )

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high confidence score (> 0.7)"""
        return self.score > 0.7
