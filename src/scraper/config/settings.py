"""Configuration management using Pydantic Settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ScraperSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Service
    service_name: str = "pioneerxity-web-scraper-service"

    # gRPC
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    grpc_max_workers: int = 10

    # FastAPI (internal-only health endpoint)
    http_enable: bool = True
    http_host: str = "127.0.0.1"
    http_port: int = 8000

    # Database (async SQLAlchemy URL)
    database_url: str = "postgresql+asyncpg://devuser:devpassword@localhost:5432/scrapper-dev"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "scraped-documents"
    minio_secure: bool = False

    # Scraping constraints (system-level)
    scrape_timeout_ms: int = 30000
    scrape_user_agent: str = "WebScraperService/0.1.0"
    max_total_documents_per_job: int = 5000
    scrape_max_retries: int = 2
    scrape_retry_backoff_ms: int = 500

    # Document size safety (generated output size)
    max_document_bytes: int = 25 * 1024 * 1024  # 25MB

    # Image handling (system-level safety)
    max_images_per_document: int = 50
    max_image_bytes: int = 5 * 1024 * 1024

    # Content filtering defaults
    default_noise_selectors: list[str] = [
        "header",
        "footer",
        "nav",
        ".navigation",
        ".sidebar",
        ".menu",
        ".breadcrumb",
        ".search",
        ".social-media",
        ".advertisement",
    ]

    # Content quality thresholds (system-level)
    min_word_count: int = 100
    min_text_to_html_ratio: float = 0.05

    # Phase 2 discovery-time quality thresholds (more permissive than Phase 1).
    # Discovery primarily needs URLs + hierarchy; strict content quality is enforced again in Phase 1.
    discovery_min_word_count: int = 10
    discovery_min_text_to_html_ratio: float = 0.0

    # Rate limiting (optional)
    enable_rate_limiting: bool = False
    rate_limit_per_domain_rps: float = 2.0

    # robots.txt (politeness)
    respect_robots_txt: bool = True
    robots_cache_ttl_seconds: int = 3600
    robots_timeout_seconds: int = 10

    # ---------------------------
    # Phase 2: LLM (Ollama) + Discovery constraints (system-level caps)
    # ---------------------------
    llm_provider: str = "ollama"  # "ollama" | "openai"
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_default_model: str = "qwen2.5:3b"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_default_model: str = "gpt-4o"

    # Phase 3: Agent backend selection
    # "heuristic" (default) or "browser_use"
    phase3_agent_backend: str = "heuristic"
    browser_use_service_base_url: str = "http://localhost:9010"
    browser_use_internal_api_key: str | None = None
    browser_use_timeout_seconds: int = 600
    browser_use_max_steps: int = 60
    browser_use_temperature: float = 0.2
    browser_use_headless: bool = True

    discovery_max_depth_cap: int = 5
    # Default high enough to cover "scrape all docs" for typical knowledge bases (e.g. CYON ~600 pages),
    # while still bounded for safety. Can be tightened in production via env DISCOVERY_MAX_PAGES_CAP.
    discovery_max_pages_cap: int = 2000

    llm_temperature_default: float = 0.1
    llm_max_tokens_cap: int = 4096
    llm_timeout_seconds_cap: int = 120

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def validate(self) -> None:
        if self.grpc_port <= 0:
            raise ValueError("grpc_port must be > 0")
        if self.http_port <= 0:
            raise ValueError("http_port must be > 0")
        if self.scrape_timeout_ms <= 0:
            raise ValueError("scrape_timeout_ms must be > 0")
        if self.max_total_documents_per_job <= 0:
            raise ValueError("max_total_documents_per_job must be > 0")
        if self.scrape_max_retries < 0:
            raise ValueError("scrape_max_retries must be >= 0")
        if self.scrape_retry_backoff_ms < 0:
            raise ValueError("scrape_retry_backoff_ms must be >= 0")
        if self.max_document_bytes <= 0:
            raise ValueError("max_document_bytes must be > 0")
        if self.max_images_per_document <= 0:
            raise ValueError("max_images_per_document must be > 0")
        if self.max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be > 0")
        if self.min_word_count <= 0:
            raise ValueError("min_word_count must be > 0")
        if self.min_text_to_html_ratio <= 0:
            raise ValueError("min_text_to_html_ratio must be > 0")
        if self.robots_cache_ttl_seconds <= 0:
            raise ValueError("robots_cache_ttl_seconds must be > 0")
        if self.robots_timeout_seconds <= 0:
            raise ValueError("robots_timeout_seconds must be > 0")
        if self.discovery_max_depth_cap <= 0:
            raise ValueError("discovery_max_depth_cap must be > 0")
        if self.discovery_max_pages_cap <= 0:
            raise ValueError("discovery_max_pages_cap must be > 0")
        if self.llm_max_tokens_cap <= 0:
            raise ValueError("llm_max_tokens_cap must be > 0")
        if self.llm_timeout_seconds_cap <= 0:
            raise ValueError("llm_timeout_seconds_cap must be > 0")
        if self.browser_use_timeout_seconds <= 0:
            raise ValueError("browser_use_timeout_seconds must be > 0")
        if self.browser_use_max_steps <= 0:
            raise ValueError("browser_use_max_steps must be > 0")


_settings: ScraperSettings | None = None


def get_settings() -> ScraperSettings:
    global _settings
    if _settings is None:
        _settings = ScraperSettings()
        _settings.validate()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None


