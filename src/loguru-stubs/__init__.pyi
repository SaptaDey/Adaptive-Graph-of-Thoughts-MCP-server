from typing import Any, Callable, TypeVar, Union, overload

# No need to import unused modules

_T = TypeVar("_T")

class Logger:
    def debug(self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    def critical(self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, __message: str, *args: Any, **kwargs: Any) -> None: ...
    def log(self, __level: int, __message: str, *args: Any, **kwargs: Any) -> None: """
Logs a message at the specified integer log level.

Args:
    __level: The severity level as an integer.
    __message: The message to log.
"""
...
    def remove(self, __handler_id: int = ...) -> None: """
Removes a logging handler by its handler ID.

Args:
    __handler_id: The integer identifier of the handler to remove.
"""
...
    def add(self, sink: Any, **kwargs: Any) -> int: """
Adds a logging sink with optional configuration and returns its handler ID.

Args:
	sink: The destination for log messages, such as a file, stream, or callable.

Returns:
	The integer handler ID assigned to the added sink.
"""
...

    @overload
    def catch(self, function: Callable[..., _T]) -> Callable[..., _T]: """
Decorator that wraps a function to automatically catch and log exceptions.

Returns:
    A wrapped function that logs exceptions raised during execution.
"""
...

    @overload
    def catch(
        self,
        exception: Union[type[BaseException], tuple[type[BaseException], ...]] = ...,
        *,
        level: str = ...,
        reraise: bool = ...,
        message: str = ...,
        onerror: Callable[..., Any] = ...,
    ) -> Callable[[Callable[..., _T]], Callable[..., _T]]: """
        Creates a decorator that wraps a function to catch specified exceptions and log them.
        
        Args:
        	exception: Exception type or tuple of exception types to catch.
        	level: Log level to use when logging the exception.
        	reraise: Whether to re-raise the exception after logging.
        	message: Message to log when an exception is caught.
        	onerror: Callable to execute if an exception is caught.
        
        Returns:
        	A decorator that wraps a function, logging and optionally handling exceptions according to the provided parameters.
        """
        ...

# Removed the non-overload decorated 'catch' method for stub file compliance
# def catch(self, *args: Any, **kwargs: Any) -> Any: ...

logger: Logger = ...

def configure(**kwargs: Any) -> None: """
Configures the logging system with the provided keyword arguments.

Args:
	**kwargs: Arbitrary configuration options for the logging system.
"""
...
