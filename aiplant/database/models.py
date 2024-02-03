from typing import NamedTuple


class _DatabaseEntry(NamedTuple):
    """A database entry."""

    datetime_utc: int
    temperature: float
    humidity: float
    target: bool
