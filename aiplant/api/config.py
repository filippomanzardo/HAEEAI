import os
from enum import StrEnum

from pydantic import BaseSettings, Field

_ENV_FILE = f"./{os.environ.get('ENVIRONMENT', 'development')}.env"


class Environment(StrEnum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class ApiSettings(BaseSettings):
    class Config:
        env_file = _ENV_FILE
        env_file_encoding = "utf-8"

    api_port: int = Field(5000, description="The port to run the API on.")

    cleanup_timeout: int = Field(
        10, description="The timeout in seconds to wait for the dependencies to stop."
    )

    environment: Environment = Field(
        Environment.DEVELOPMENT,
        description="The environment to run the API in.",
    )
