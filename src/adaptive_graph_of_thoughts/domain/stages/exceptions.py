class StageError(Exception):
    """Base class for stage-related errors."""


class StageInitializationError(StageError):
    """Raised when a stage fails to initialize required resources."""

