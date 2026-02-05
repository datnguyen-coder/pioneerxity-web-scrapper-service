#!/usr/bin/env python3
"""Phase 2 test client: DiscoverStructure -> GetDiscoveredStructure -> ApproveAndScrape.

Requires:
- Web Scraper Service running (gRPC)
- Ollama running (default localhost:11434) with model qwen2.5:3b (or override)
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

import grpc

try:
    from proto.generated import scraper_pb2, scraper_pb2_grpc
except ImportError:
    print("ERROR: gRPC code not generated. Run: uv run python scripts/generate_proto.py")
    sys.exit(1)


async def discover(host: str, port: int, *, base_url: str, model: str, max_depth: int, max_pages: int) -> str:
    address = f"{host}:{port}"
    async with grpc.aio.insecure_channel(address) as channel:
        stub = scraper_pb2_grpc.WebScraperServiceStub(channel)
        job_id = str(uuid.uuid4())

        req = scraper_pb2.AutonomousScrapeRequest(
            job_id=job_id,
            client_id="test-client-phase2",
            base_url=base_url,
            discovery_options=scraper_pb2.DiscoveryOptions(max_depth=max_depth, max_pages=max_pages),
            llm_options=scraper_pb2.LLMOptions(
                provider="ollama",
                model=model,
                temperature=0.1,
                max_tokens=1024,
                timeout_seconds=60,
            ),
            scraping_options=scraper_pb2.ScrapingOptions(
                output_format="pdf",
                include_images=False,
                image_handling="embed_or_link",
                noise_selectors=["header", "footer", "nav"],
                include_metadata=True,
            ),
        )

        print(f"Connecting to {address}")
        print(f"Discovery Job ID: {job_id}")
        print("Streaming discovery progress...\n")

        async for progress in stub.DiscoverStructure(req):
            print(
                f"[{progress.stage}] crawled={progress.pages_crawled}/{progress.max_pages} "
                f"complete={progress.is_complete} success={progress.success} url={progress.current_url} "
                f"msg={progress.message}"
            )
            if progress.error_message:
                print(f"  error_code={progress.error_code} error={progress.error_message}")
            if progress.discovered_structure_json:
                print("  discovered_structure_json=<present>")
            if progress.is_complete:
                break

        return job_id


async def get_discovered(host: str, port: int, *, job_id: str) -> dict:
    address = f"{host}:{port}"
    async with grpc.aio.insecure_channel(address) as channel:
        stub = scraper_pb2_grpc.WebScraperServiceStub(channel)
        resp = await stub.GetDiscoveredStructure(scraper_pb2.GetDiscoveredStructureRequest(job_id=job_id))
        if not resp.success:
            raise RuntimeError(resp.error_message or "GetDiscoveredStructure failed")
        print("\nDiscovered structure (JSON):")
        print(resp.discovered_structure_json)
        if getattr(resp, "scraping_options_json", ""):
            print("\nMerged scraping options (JSON):")
            print(resp.scraping_options_json)
        return json.loads(resp.discovered_structure_json)


async def approve_and_scrape(host: str, port: int, *, job_id: str, approved_structure: dict | None = None) -> None:
    address = f"{host}:{port}"
    async with grpc.aio.insecure_channel(address) as channel:
        stub = scraper_pb2_grpc.WebScraperServiceStub(channel)
        req = scraper_pb2.ApproveAndScrapeRequest(
            job_id=job_id,
            approved_structure_json=json.dumps(approved_structure, ensure_ascii=False) if approved_structure else "",
        )

        print("\nStreaming scrape progress...\n")
        async for progress in stub.ApproveAndScrape(req):
            print(
                f"[{progress.stage}] idx={progress.current_index}/{progress.total_documents} "
                f"complete={progress.is_complete} success={progress.success} url={progress.current_url} "
                f"msg={progress.message}"
            )
            if progress.error_message:
                print(f"  error_code={progress.error_code} error={progress.error_message}")
            if progress.last_document and progress.last_document.minio_path:
                print(f"  uploaded={progress.last_document.minio_path}")
            if progress.is_complete:
                if progress.stats:
                    print(
                        f"\nDone. total={progress.stats.total_pages_processed} "
                        f"ok={progress.stats.successful_pages} failed={progress.stats.failed_pages} "
                        f"ms={progress.stats.total_processing_time_ms}"
                    )
                break


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--base-url", required=True, help="Docs base URL to discover (e.g. https://www.cyon.ch/support)")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--llm-timeout", type=int, default=180, help="LLM timeout seconds (Ollama)")
    parser.add_argument("--llm-max-tokens", type=int, default=1024, help="LLM max tokens (Ollama num_predict)")
    parser.add_argument("--output-format", choices=["pdf", "docx", "both"], default="pdf")
    parser.add_argument("--include-images", action="store_true", help="Enable image scraping for ApproveAndScrape")
    parser.add_argument(
        "--image-handling",
        choices=["embed", "link_only", "both", "embed_or_link"],
        default="embed_or_link",
        help="How to handle images in generated documents",
    )
    parser.add_argument("--min-word-count", type=int, default=0, help="Phase 2 quality gate (0 = use server default)")
    parser.add_argument(
        "--min-text-to-html-ratio",
        type=float,
        default=0.0,
        help="Phase 2 quality gate (0 = use server default)",
    )
    args = parser.parse_args()

    # Override the module-level defaults by rebuilding request with args in discover().
    # (Keep the client simple and deterministic.)
    async def _discover_with_args() -> str:
        address = f"{args.host}:{args.port}"
        async with grpc.aio.insecure_channel(address) as channel:
            stub = scraper_pb2_grpc.WebScraperServiceStub(channel)
            job_id = str(uuid.uuid4())

            req = scraper_pb2.AutonomousScrapeRequest(
                job_id=job_id,
                client_id="test-client-phase2",
                base_url=args.base_url,
                discovery_options=scraper_pb2.DiscoveryOptions(
                    max_depth=args.max_depth,
                    max_pages=args.max_pages,
                    quality_thresholds=scraper_pb2.QualityThresholds(
                        min_word_count=args.min_word_count,
                        min_text_to_html_ratio=args.min_text_to_html_ratio,
                    )
                    if (args.min_word_count > 0 or args.min_text_to_html_ratio > 0.0)
                    else None,
                ),
                llm_options=scraper_pb2.LLMOptions(
                    provider="ollama",
                    model=args.model,
                    temperature=0.1,
                    max_tokens=args.llm_max_tokens,
                    timeout_seconds=args.llm_timeout,
                ),
                scraping_options=scraper_pb2.ScrapingOptions(
                    output_format=args.output_format,
                    include_images=bool(args.include_images),
                    image_handling=args.image_handling,
                    # Broader noise filtering to reduce boilerplate.
                    noise_selectors=[
                        "header",
                        "footer",
                        "nav",
                        "aside",
                        ".navigation",
                        ".sidebar",
                        ".menu",
                        ".breadcrumb",
                        ".search",
                        ".social-media",
                        ".advertisement",
                        ".cookie",
                        ".cookie-banner",
                    ],
                    include_metadata=True,
                ),
            )

            print(f"Connecting to {address}")
            print(f"Discovery Job ID: {job_id}")
            print("Streaming discovery progress...\n")

            async for progress in stub.DiscoverStructure(req):
                print(
                    f"[{progress.stage}] crawled={progress.pages_crawled}/{progress.max_pages} "
                    f"complete={progress.is_complete} success={progress.success} url={progress.current_url} "
                    f"msg={progress.message}"
                )
                if progress.error_message:
                    print(f"  error_code={progress.error_code} error={progress.error_message}")
                if progress.discovered_structure_json:
                    print("  discovered_structure_json=<present>")
                if progress.is_complete:
                    break
            return job_id

    job_id = await _discover_with_args()
    structure = await get_discovered(args.host, args.port, job_id=job_id)
    await approve_and_scrape(args.host, args.port, job_id=job_id, approved_structure=structure)


if __name__ == "__main__":
    asyncio.run(main())


