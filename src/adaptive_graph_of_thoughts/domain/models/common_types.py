"""
Common type definitions to avoid circular imports.
Provides type definitions used across multiple modules.
"""

from pydantic import BaseModel, Field


class GoTProcessorSessionData(BaseModel):
    """Data model for session data maintained by GoTProcessor."""

    session_id: str = Field(default="")
    query: str = ""
    final_answer: str = ""
    final_confidence_vector: str = "0.5,0.5,0.5,0.5"  # Simplified as string
    accumulated_context: str = ""  # Simplified as JSON string
    stage_outputs_trace: str = ""  # Simplified as JSON string


# ASRGoTGraph import is not present in this file, so no removal needed here for that.
# If ASRGoTGraph was imported for typing graph_state, that line would also be removed.


class ComposedOutput(BaseModel):
    """Model for the output structure from the Composition Stage."""

    executive_summary: str = ""
    detailed_report: str = ""
    key_findings: str = ""  # Simplified as string instead of list
    confidence_assessment: str = ""  # Simplified as JSON string
