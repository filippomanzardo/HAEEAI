from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from aiplant.database.models import _DatabaseEntry

Target = NewType("Label", bool)


@dataclass(frozen=True)
class Feature:
    """A labeled entry."""

    timestamp: int
    """The timestamp of the entry."""

    temperature: float
    """The temperature recorded."""

    moisture: float
    """The moisture recorded."""

    @classmethod
    def from_database_entry(cls, entry: _DatabaseEntry) -> Feature:
        """Create an Input from a database entry."""
        return cls(
            timestamp=entry.datetime_utc,
            temperature=entry.temperature,
            moisture=entry.humidity,
        )

    def to_tuple(self) -> tuple[int, float, float]:
        """Return the feature as a tuple."""
        return (self.timestamp, self.temperature, self.moisture)


Sample = NewType("Sample", tuple[Feature, Target])


def to_sample(entry: _DatabaseEntry) -> Sample:
    """Create a Sample from a database entry."""
    return Sample((Feature.from_database_entry(entry), entry.target))
