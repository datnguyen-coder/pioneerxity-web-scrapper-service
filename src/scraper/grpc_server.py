"""gRPC server setup and configuration."""

from __future__ import annotations

from concurrent import futures

import grpc

from .config.settings import get_settings
from .controllers.scraper_controller import WebScraperController
from .observability.logger import get_logger

# Global app state populated during lifespan startup
app_state: dict = {}

# Import generated gRPC code
try:
    from proto.generated import scraper_pb2_grpc
except ImportError as e:
    raise ImportError(
        "gRPC code not generated. Run: python scripts/generate_proto.py"
    ) from e


async def serve() -> None:
    """Start the gRPC server."""
    settings = get_settings()
    logger = get_logger(__name__)

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers))

    scraper_pb2_grpc.add_WebScraperServiceServicer_to_server(
        WebScraperController(
            app_state["scraper_service"],
            app_state["discovery_service"],
            app_state["agent_service"],
        ),
        server,
    )

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    await server.start()
    logger.info("grpc_server_started", address=listen_addr)
    await server.wait_for_termination()


