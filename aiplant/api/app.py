import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from aiplant.api.config import ApiSettings
from aiplant.api.dependencies import api_lifespan, setup_dependencies

_LOGGER = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create the Server API instance.

    :return: the API server instance
    """

    api_config = ApiSettings()
    dependencies = setup_dependencies()

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

    return app
