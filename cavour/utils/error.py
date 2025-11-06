"""
Custom exception class for Cavour library errors.

Provides a specialized exception type to distinguish errors originating
from the Cavour library from other Python exceptions.

Example:
    >>> from cavour.utils.error import LibError
    >>>
    >>> # Raise library-specific error
    >>> if value < 0:
    ...     raise LibError("Value must be non-negative")
    >>>
    >>> # Catch library errors specifically
    >>> try:
    ...     curve.df(invalid_date)
    ... except LibError as e:
    ...     print(f"Cavour error: {e._message}")
"""

class LibError(Exception):
    """ Class to understand if the error is coming from this library """

    def __init__(self,
                 message: str):
        """ Create error object """
        self._message = message

    def _print(self):
        print("LibError:",self._message)
