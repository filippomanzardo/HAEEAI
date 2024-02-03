from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse

from aiplant.database.eeprom import EEPROMDatabase, PlantId
from aiplant.model.models import Feature
from aiplant.model.waterer import Waterer


def create_ai_plant_router(database: EEPROMDatabase, waterer: Waterer) -> APIRouter:
    """Build a FastAPI router for serving aiPlant endpoints."""
    router = APIRouter()

    @router.get("/{plant_id}")
    async def get_plant_data(plant_id: PlantId) -> Feature:
        """Return the latest data for the plant."""
        now = int(datetime.now().timestamp())
        latest_entry = await database.get_latest_entry(plant_id=plant_id, timestamp=now)

        if latest_entry is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for the plant. Please retry later.",
            )

        return Feature.from_database_entry(latest_entry)

    @router.get(
        "/{plant_id}/water",
        response_model=bool,
    )
    async def should_you_water_plan(plant_id: PlantId) -> bool:
        """Return if you should water the plant."""
        now = int(datetime.now().timestamp())
        latest_entry = await database.get_latest_entry(plant_id=plant_id, timestamp=now)

        if latest_entry is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data found for the plant. Please retry later.",
            )

        feature = Feature.from_database_entry(latest_entry)
        return await waterer.predict(feature)

    return router
