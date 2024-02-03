import logging

from fastapi import FastAPI

from aiplant.api.config import ApiSettings
from aiplant.api.dependencies import api_lifespan, setup_dependencies
from aiplant.api.routers import aiplant, model

_LOGGER = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create the Server API instance.

    :return: the API server instance
    """

    api_config = ApiSettings()
    dependencies = setup_dependencies()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = FastAPI(
        description="The aiPlant backend.",
        docs_url="/-/docs/swagger",
        openapi_url="/-/docs/openapi.json",
        redoc_url="/-/docs/docs/redoc",
        title=__name__,
        lifespan=api_lifespan,
    )

    app.state.dependencies = dependencies
    app.state.api_config = api_config

    app.include_router(
        router=aiplant.create_ai_plant_router(
            database=dependencies.database,
            waterer=dependencies.waterer,
        ),
        prefix="/plants",
        tags=["API"],
    )

    app.include_router(
        router=model.create_model_router(
            waterer=dependencies.waterer,
            database=dependencies.database,
            labeler=dependencies.labeler,
            adapter=dependencies.bluetooth_adapter,
        ),
        prefix="/model",
        tags=["API"],
    )

    return app
