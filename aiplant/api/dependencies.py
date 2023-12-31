import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from aiplant import EEPROMDatabase
from aiplant.api.config import ApiSettings

_LOGGER = logging.getLogger(__name__)


@dataclass
class Dependencies:
    database: EEPROMDatabase

    async def start_up(self) -> None:
        """Initialize the dependencies."""
        await asyncio.gather(
            self.database.connect(),
        )

    async def stop(self, timeout_seconds: int) -> None:
        """Stop the dependencies."""
        await asyncio.wait_for(
            asyncio.gather(self.database.disconnect()), timeout=timeout_seconds
        )


def setup_dependencies() -> Dependencies:
    """Setup the dependencies."""
    database = EEPROMDatabase(database_path="./database/environment.csv")
    return Dependencies(
        database=database,
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
