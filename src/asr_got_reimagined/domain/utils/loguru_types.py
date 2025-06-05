"""
Type stubs for loguru to help with static type checking.
This file provides type annotations for the loguru module to suppress import and type errors.
"""

from typing import (
    Any,
    Callable,
    TypeVar,
    overload,
)  # Removed List

T = TypeVar("T")  # Defined T


class Logger:
    """Type stub for loguru.Logger class."""

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log info level message."""
        ...

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log debug level message."""
        ...

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning level message."""
        ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error level message."""
        ...

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log critical level message."""
        ...

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log exception with traceback."""
        ...

    def remove(self, handler_id: int) -> None:
        """Remove a previously added handler."""
        ...

    def add(self, sink: Any, **kwargs: Any) -> int:
        """
        Adds a new logging handler to the logger.
        
        Args:
            sink: The destination for log messages, such as a file, stream, or callable.
            **kwargs: Additional configuration options for the handler.
        
        Returns:
            An integer identifier for the added handler.
        """
        ...

    @overload
    def catch(
        self, exception: Callable[..., T]
    ) -> Callable[..., T]:  # T should be defined now
        """
        Decorator that wraps a callable to automatically catch and log exceptions.
        
        Returns:
            A callable with the same signature as the input, with exception handling added.
        """
        ...  # type: ignore[empty-body]

    @overload
    def catch(
        self,
        exception: list[Exception] = ...,
        *,  # Changed List to list
        default: Any = ...,
        message: str = ...,
        onerror: Callable[..., Any] = ...,
        level: str = ...,
        reraise: bool = ...,
        exclude: list[Exception] = ...,
    ) -> Callable[
        [Callable[..., T]], Callable[..., T]
    ]:  # Changed List to list, T should be defined
        """
        Decorator that wraps a function to automatically catch and log specified exceptions.
        
        Args:
            exception: List of exception types to catch. Defaults to all exceptions.
            default: Value to return if an exception is caught.
            message: Optional log message to use when an exception is caught.
            onerror: Optional callback invoked when an exception is caught.
            level: Log level to use for the caught exception.
            reraise: If True, re-raises the exception after logging.
            exclude: List of exception types to ignore and not catch.
        
        Returns:
            A decorator that wraps a function, catching and logging specified exceptions.
        """
        ...  # type: ignore[empty-body]

    def catch(self, *args: Any, **kwargs: Any) -> Any:
        """
        Decorator that catches exceptions raised by a function and logs them.
        
        Returns:
            The decorated function with exception handling and logging applied.
        """
        ...  # type: ignore[empty-body]


# Create a global logger instance
logger = Logger()
