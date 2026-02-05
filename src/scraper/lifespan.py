"""Application lifespan management (startup/shutdown hooks)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from .config.settings import get_settings
from .observability.logger import configure_logging, get_logger
from .storage.minio_client import MinIOClient
from .storage.database import close_db, init_db, session_factory
from .storage.repositories import DocumentRepository, JobRepository
from .storage.repositories import DiscoveryRepository
from .services.content_filter import ContentFilter
from .services.document_processor import DocumentProcessor
from .services.discovery_service import DiscoveryService
from .services.agent_service import AgentScraperService
from .services.scraper_service import WebScraperService
from .scraping.playwright_scraper import PlaywrightScraper
from .processing.image_handler import ImageHandler
from .utils.rate_limiter import DomainRateLimiter
from .utils.robots import RobotsChecker
from .llm.ollama_adapter import OllamaAdapter

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan_manager():
    """Manage application lifespan (startup and shutdown)."""
    settings = get_settings()

    configure_logging()
    logger.info("starting_application", service_name=settings.service_name)

    # Initialize database (create tables for MVP)
    await init_db()
    logger.info("database_initialized")

    # Initialize MinIO client and validate connectivity
    minio = MinIOClient(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
        bucket=settings.minio_bucket,
    )
    if await minio.health_check():
        logger.info("minio_healthy", endpoint=settings.minio_endpoint, bucket=settings.minio_bucket)
    else:
        logger.warning("minio_health_check_failed", endpoint=settings.minio_endpoint)

    # Build layer dependencies (strict separation)
    rate_limiter = None
    if settings.enable_rate_limiting:
        rate_limiter = DomainRateLimiter(requests_per_second=settings.rate_limit_per_domain_rps)
        logger.info(
            "rate_limiter_enabled",
            rate_limit_per_domain_rps=settings.rate_limit_per_domain_rps,
        )

    robots_checker = RobotsChecker(
        cache_ttl_seconds=settings.robots_cache_ttl_seconds,
        timeout_seconds=settings.robots_timeout_seconds,
    )
    logger.info(
        "robots_checker_configured",
        respect_robots_txt=settings.respect_robots_txt,
        robots_cache_ttl_seconds=settings.robots_cache_ttl_seconds,
        robots_timeout_seconds=settings.robots_timeout_seconds,
    )

    scraper = PlaywrightScraper(
        default_timeout_ms=settings.scrape_timeout_ms,
        user_agent=settings.scrape_user_agent,
        rate_limiter=rate_limiter,
        robots_checker=robots_checker,
        respect_robots_txt=settings.respect_robots_txt,
    )
    content_filter = ContentFilter(default_noise_selectors=settings.default_noise_selectors)
    document_processor = DocumentProcessor()
    image_handler = ImageHandler(
        max_images=settings.max_images_per_document,
        max_bytes=settings.max_image_bytes,
    )

    # Persistence repositories (created with a long-lived session factory; sessions per operation)
    jobs = JobRepository(session_factory=session_factory)
    documents = DocumentRepository(session_factory=session_factory)
    discovery_repo = DiscoveryRepository(session_factory=session_factory)

    # Create orchestration service
    from .grpc_server import app_state

    app_state["scraper_service"] = WebScraperService(
        scraper=scraper,
        content_filter=content_filter,
        document_processor=document_processor,
        image_handler=image_handler,
        minio=minio,
        jobs=jobs,
        documents=documents,
    )

    # Phase 2 discovery service (Ollama only for now)
    app_state["discovery_service"] = DiscoveryService(
        scraper=scraper,
        llm=OllamaAdapter(host=settings.ollama_host, port=settings.ollama_port),
        discovery_repo=discovery_repo,
    )

    # Phase 3 agent service (LLM-guided navigation)
    app_state["agent_service"] = AgentScraperService(
        scraper=scraper,
        web_scraper=app_state["scraper_service"],
    )

    logger.info("application_started")
    try:
        yield
    finally:
        await close_db()
        logger.info("application_shutdown_complete")


