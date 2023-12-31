from typing import NamedTuple

import aiofiles


class _DatabaseEntry(NamedTuple):
    """A database entry."""

    datetime_utc: int
    temperature: float
    humidity: float


class EEPROMDatabase:
    """
    A class to represent the EEPROM database.

    The database is a CSV file with the following columns:
    - datetime_utc: The datetime in UTC when the data was recorded.
    - temperature: The temperature in Celsius (> 0).
    - humidity: The humidity in percentage as basis points.
    """

    def __init__(self, database_path: str) -> None:
        self._database_path = database_path
        self._database: dict[int, _DatabaseEntry] | None = None

    @property
    def database(self) -> dict[int, _DatabaseEntry]:
        """Return the database."""
        if self._database is None:
            raise RuntimeError("Database wasn't initialized.")
        return self._database

    async def connect(self) -> None:
        """Connect to the database."""
        self._database = {}
        await self.refresh()

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        self._database = None

    async def refresh(self) -> None:
        """Refresh the database."""
        async with aiofiles.open(self._database_path, mode="r") as file:
            async for line in file:
                if line.startswith("#"):
                    continue
                timestamp, temperature, humidity = line.split(",")
                self._database[int(timestamp)] = _DatabaseEntry(
                    temperature=int(temperature),
                    humidity=int(humidity),
                    datetime_utc=int(timestamp),
                )

    async def entries(self) -> list[_DatabaseEntry]:
        """Return the database entries."""
        return list(self.database.values())

    async def get_entry(self, timestamp: int) -> _DatabaseEntry | None:
        """Return the database entry for the given timestamp."""
        return self.database.get(timestamp)
