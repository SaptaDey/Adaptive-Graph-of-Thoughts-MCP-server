from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger  # type: ignore
from pydantic import BaseModel, Field

from asr_got_reimagined.config import (
    Settings,  # To access ASRGoTDefaultParams if needed by stages
)
from asr_got_reimagined.domain.models.common_types import GoTProcessorSessionData
# ASRGoTGraph has been removed as part of refactoring


class StageOutput(BaseModel):
    """Standard output structure for each stage."""

    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    # Optional: Data to be passed to the next stage or stored in the session context
    # This can be more specific in subclasses if needed.
    next_stage_context_update: dict[str, Any] = Field(default_factory=dict)


class BaseStage(ABC):
    """Abstract Base Class for all stages in the ASR-GoT pipeline."""

    stage_name: str = "UnknownStage"  # Override in subclasses

    def __init__(self, settings: Settings):
        self.settings = settings
        self.default_params = settings.asr_got.default_parameters
        # Log initialization
        logger.debug(f"Initialized stage: {self.stage_name}")  # type: ignore

    @abstractmethod
    async def execute(
        self,
        current_session_data: GoTProcessorSessionData,  # Proper import instead of forward reference
    ) -> StageOutput:
        """
        Executes the logic for this stage.

        Args:
            current_session_data: Contains all accumulated data for the current session,
                                  including initial query, parameters, and outputs from previous stages.

        Returns:
            A StageOutput object containing a summary, metrics, and any data to update
            the session context for subsequent stages.
        """
        # This is an abstract method - concrete implementations must return a StageOutput
        raise NotImplementedError("Subclasses must implement execute method")

    def _log_start(self, session_id: Optional[str]):
        logger.info(
            f"[Session: {session_id or 'N/A'}] >>> Executing Stage: {self.stage_name} >>>"
        )

    def _log_end(self, session_id: Optional[str], output: StageOutput):
        logger.info(
            f"[Session: {session_id or 'N/A'}] <<< Completed Stage: {self.stage_name} | Summary: {output.summary} | Metrics: {output.metrics} <<<"
        )
