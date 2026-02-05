"""Application entrypoint (FastAPI + gRPC)."""

import asyncio
import contextlib
import sys

from .grpc_server import serve
from .http_server import run_http_server
from .lifespan import lifespan_manager
from .config.settings import get_settings


async def main() -> None:
    """Main application entrypoint."""
    async with lifespan_manager():
        settings = get_settings()
        http_task = None
        if settings.http_enable:
            http_task = asyncio.create_task(run_http_server())
        try:
            await serve()
        finally:
            if http_task:
                http_task.cancel()
                with contextlib.suppress(Exception):
                    await http_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


