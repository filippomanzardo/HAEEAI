import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from aiplant.api.config import ApiSettings
from aiplant.bluetooth.adapter import BluetoothAdapter
from aiplant.database.eeprom import EEPROMDatabase
from aiplant.model.labeler import Labeler
from aiplant.model.models import Sample, to_sample
from aiplant.model.waterer import Waterer

_LOGGER = logging.getLogger(__name__)


@dataclass
class Dependencies:
    bluetooth_adapter: BluetoothAdapter
    database: EEPROMDatabase
    waterer: Waterer
    labeler: Labeler

    async def start_up(self) -> None:
        """Initialize the dependencies."""
        # No asyncio.gather unfortunately
        await self.bluetooth_adapter.connect()
        await self.database.connect()

        self.labeler.add_base_samples(
            [to_sample(entry) for entry in self.database.entries]
        )

        await asyncio.gather(
            self.labeler.refresh_model(),
            self.waterer.train(),
        )

    async def stop(self, timeout_seconds: int) -> None:
        """Stop the dependencies."""
        await asyncio.wait_for(
            asyncio.gather(
                self.database.disconnect(), self.bluetooth_adapter.disconnect()
            ),
            timeout=timeout_seconds,
        )


def setup_dependencies() -> Dependencies:
    """Setup the dependencies."""
    bluetooth_adapter = BluetoothAdapter()
    database = EEPROMDatabase(
        database_path="./data/environment.csv", real_time_data=bluetooth_adapter
    )
    waterer = Waterer(database=database)
    labeler = Labeler(base_samples=[])

    return Dependencies(
        database=database,
        waterer=waterer,
        bluetooth_adapter=bluetooth_adapter,
        labeler=labeler,
    )


@asynccontextmanager
async def api_lifespan(app: FastAPI) -> Any:
    """The FastAPI lifespan context manager."""
    config: ApiSettings = app.state.api_config
    dependencies: Dependencies = app.state.dependencies
    _LOGGER.info("⚙️ About to start the dependencies... ⚙️")

    await dependencies.start_up()

    _LOGGER.info("✅ Dependencies ready ✅")

    yield

    _LOGGER.info("⚙️ About to stop the dependencies... ⚙️")

    await dependencies.stop(timeout_seconds=config.cleanup_timeout)

    _LOGGER.info("🏁 Dependencies stopped 🏁")
