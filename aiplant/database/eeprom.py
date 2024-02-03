import asyncio
import logging
import random
from typing import NewType

import aiofiles
from cachetools import TTLCache

from aiplant.bluetooth.adapter import BluetoothAdapter
from aiplant.database.models import _DatabaseEntry
from aiplant.model.models import Feature, Sample

_LOGGER = logging.getLogger(__name__)

PlantId = NewType("PlantId", int)
Key = NewType("Key", tuple[int, PlantId])
_MAX_SIZE = 3_000_000
_TTL = 60 * 60 * 24 * 7  # 1 week


class EEPROMDatabase:
    """
    A class to represent the EEPROM database.

    The database is a CSV file with the following columns:
    - datetime_utc: The datetime in UTC when the data was recorded.
    - temperature: The temperature in Celsius (> 0).
    - humidity: The humidity in percentage as basis points.
    """

    def __init__(self, database_path: str, real_time_data: BluetoothAdapter) -> None:
        self._database_path = database_path
        self._database: dict[Key, _DatabaseEntry] | None = None
        self._real_time_samples = TTLCache(maxsize=3_000_000, ttl=_TTL)
        self._real_time_data = real_time_data

    @property
    def database(self) -> dict[Key, _DatabaseEntry]:
        """Return the database."""
        if self._database is None:
            raise RuntimeError("Database wasn't initialized.")
        return self._database

    def load_samples(self, samples: list[Sample], plat_id: PlantId = PlantId(1)) -> None:
        """Load the samples into the database."""
        for sample in samples:
            key = Key((sample[0].timestamp, plat_id))
            self._database[key] = _DatabaseEntry(
                temperature=sample[0].temperature,
                humidity=sample[0].moisture,
                datetime_utc=sample[0].timestamp,
                target=sample[1],
            )


    @property
    def real_time_samples(self) -> list[Feature]:
        """Return the real-time samples."""
        return [
            Feature(
                timestamp=key,
                moisture=value[0],
                temperature=value[1],
            )
            for key, value in self._real_time_samples.items()
        ]


    async def connect(self) -> None:
        """Connect to the database."""
        self._database = {}
        await self.refresh()

        async def _loop_tasks() -> None:
            while True:
                feature = await self._real_time_data.get_feature()
                self._real_time_samples[feature.timestamp] = (
                    feature.moisture,
                    feature.temperature,
                )
                await asyncio.sleep(1)

        loop = asyncio.get_event_loop()
        asyncio.ensure_future(_loop_tasks(), loop=loop)

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        self._database = None

    async def refresh(self) -> None:
        """Refresh the database."""
        _LOGGER.info("🌱 Loading the local database... 🌱")
        timestamp = 1708063280
        offset = 10
        async with aiofiles.open(self._database_path, mode="r") as file:
            async for line in file:
                temperature, humidity, plant_id, target = line.split(",")
                # Skip the header
                if not temperature.isdigit():
                    continue

                # Simulate the timestamp
                timestamp += offset

                key = Key((int(timestamp), PlantId(int(plant_id))))
                self._database[key] = _DatabaseEntry(
                    temperature=int(temperature),
                    humidity=int(humidity),
                    datetime_utc=int(timestamp),
                    target=not bool(int(target.strip())),  # Pump OFF == 1
                )

        _LOGGER.info("✅ Database loaded ✅")

    @property
    def entries(self) -> list[_DatabaseEntry]:
        """Return the database entries."""
        return list(self.database.values())

    async def get_entry(
        self, timestamp: int, plant_id: PlantId
    ) -> _DatabaseEntry | None:
        """Return the database entry for the given timestamp."""
        key = Key((timestamp, plant_id))
        return self.database.get(Key(key))

    async def get_latest_entry(
        self, plant_id: PlantId, timestamp: int
    ) -> _DatabaseEntry | None:
        """Return the latest entry for the given plant."""
        entries = [
            entry
            for key, entry in self.database.items()
            if key[1] == plant_id and key[0] <= timestamp
        ]
        return max(entries, key=lambda entry: entry.datetime_utc) if entries else None

    def log_status(self) -> None:
        num_entries = len(self._real_time_samples)
        temp_entries = [
            temp for _, temp in self._real_time_samples.values()
        ]
        moist_entries = [
            moist for moist, _ in self._real_time_samples.values()
        ]
        max_, min_, mean_ = (
            max(temp_entries),
            min(temp_entries),
            sum(temp_entries) / num_entries,
        )
        max_m, min_m, mean_m = (
            max(moist_entries),
            min(moist_entries),
            sum(moist_entries) / num_entries,
        )
        _LOGGER.info(
            "\n🌱 Real-time data has %s entries 🌱\n"
            "Last entry: %s\n"
            "Data:\ntemp(min,max,mean) -> (%.2f,%.2f,%.2f)\n"
            "moist(min,max,mean) -> (%.2f,%.2f,%.2f)\n",
            num_entries,
            list(self._real_time_samples.items())[-1],
            min_,
            max_,
            mean_,
            min_m,
            max_m,
            mean_m,
        )
