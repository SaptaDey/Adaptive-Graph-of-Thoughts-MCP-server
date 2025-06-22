"""Exceptions specific to the GoTProcessor services."""

class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass


class StageExecutionError(ProcessingError):
    """Error raised when a stage fails during execution."""

    def __init__(self, stage_name: str, original_error: Exception):
        self.stage_name = stage_name
        self.original_error = original_error
        message = f"Stage '{stage_name}' failed: {original_error}"
        super().__init__(message)
