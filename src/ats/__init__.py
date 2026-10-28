# src/ats/ — the package of "connectors": each module here talks to ONE external job-board
# service (Greenhouse, Lever, …) and returns jobs in the project's shared normalized shape.
#
# This __init__.py defines a common base error so callers can catch problems from ANY
# connector with a single `except ATSError`, instead of naming each board's error type.


class ATSError(RuntimeError):
    """Base error for any ATS/job-board connector failure (bad token, network error,
    non-200 response). Each connector raises a subclass (GreenhouseError, LeverError),
    so `except ATSError` catches problems from any board."""
