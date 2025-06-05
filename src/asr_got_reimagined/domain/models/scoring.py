"""
Scoring models for ASR-GoT system.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from asr_got_reimagined.domain.models.common import ConfidenceVector


class ScoreResult(BaseModel):
    """
    Model for scoring results in the ASR-GoT system.
    
    This model is used to represent scoring results from the GoTProcessor,
    including confidence scores, metrics, and other evaluation data.
    """
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall score value between 0 and 1")
    confidence_vector: Optional[ConfidenceVector] = Field(
        default=None, description="Confidence vector components if applicable"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metrics and measurements"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed scoring information"
    )
    category_scores: Dict[str, float] = Field(
        default_factory=dict, description="Scores broken down by category"
    )
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high confidence score (> 0.7)"""
        return self.score > 0.7
