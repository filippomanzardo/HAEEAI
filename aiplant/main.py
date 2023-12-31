import uvicorn

from aiplant.api.config import ApiSettings, Environment


def start_uvicorn_server(config: ApiSettings) -> None:
    """Start the development server."""
    host = "0.0.0.0"
    port = config.api_port
    debug = config.environment == Environment.DEVELOPMENT
    uvicorn.run(
        "aiplant.api.app:create_app",
        host=host,
        port=port,
        access_log=debug,
        reload=debug,
        factory=True,
        proxy_headers=True,
        lifespan="on",
    )


def main() -> None:
    """Entrypoint."""
    config = ApiSettings()
    start_uvicorn_server(config)


if __name__ == "__main__":
    main()
