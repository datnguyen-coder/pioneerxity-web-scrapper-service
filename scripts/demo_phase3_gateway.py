#!/usr/bin/env python3
"""
Phase 3 demo runner (Gateway-like client).

Flow (counter-clockwise, no step numbers):
User -> Gateway -> Agent Service -> Collected URLs -> Scraper Service (Phase 1) -> MinIO/Postgres -> Gateway -> User

This demo keeps "business logic" out of the Gateway:
- Gateway only forwards (base_url, prompt, constraints) to the agent collector,
  then converts the returned URL list into a Phase 1 `config_json` and streams `ScrapeConfigured`.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import uuid
from urllib.parse import urlparse

import grpc

from phase3_agent_stub import AgentConstraints, collect_document_urls

# Ensure repo root is on sys.path so `proto.generated` resolves to this repo (not any site-packages `proto`).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
for p in (str(_SRC_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from proto.generated import scraper_pb2, scraper_pb2_grpc
except ImportError:
    print("ERROR: gRPC code not generated. Run: uv run python scripts/generate_proto.py")
    sys.exit(1)


def _to_relative_paths(base_url: str, urls: list[str]) -> list[str]:
    base = urlparse(base_url)
    base_path = (base.path or "").rstrip("/")
    out: list[str] = []
    for u in urls:
        p = urlparse(u)
        if p.netloc and p.netloc != base.netloc:
            continue
        path = p.path or "/"
        # Fix common duplication when joining relative links from a trailing-slash base:
        # e.g. base "/support" + href "support/a/x" -> "/support/support/a/x"
        if base_path and path.startswith(base_path + base_path + "/"):
            # "/support/support/a/x" -> "/support/a/x"
            path = base_path + path[len(base_path + base_path) :]

        # IMPORTANT: Phase 1 expects document paths relative to `base_url`.
        # If `base_url` already contains a scope path (e.g. "/support"), and we pass "/support/a/x",
        # Phase 1 will produce "https://.../support/support/a/x". So strip the base_path prefix.
        if base_path and path.startswith(base_path + "/"):
            path = path[len(base_path) :]  # keep leading "/"
        out.append(path)
    # unique + stable
    seen = set()
    uniq = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


async def run_demo(
    *,
    host: str,
    port: int,
    base_url: str,
    prompt: str,
    max_depth: int,
    max_pages: int,
    output_format: str,
    include_images: bool,
    image_handling: str,
    llm_provider: str,
    ollama_host: str,
    ollama_port: int,
    ollama_model: str,
    openai_api_key: str | None,
    openai_base_url: str | None,
    openai_model: str,
) -> None:
    job_id = str(uuid.uuid4())

    # "Agent Service": collect relevant doc URLs using LLM-guided navigation
    constraints = AgentConstraints(
        max_depth=max_depth,
        max_pages=max_pages,
        llm_provider=llm_provider,
        ollama_host=ollama_host,
        ollama_port=ollama_port,
        ollama_model=ollama_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
    )
    print(f"\n[Gateway] Calling agent service to collect URLs...")
    print(f"[Gateway] Prompt: '{prompt}'")
    print(f"[Gateway] Base URL: {base_url}\n")
    
    collected = await collect_document_urls(base_url=base_url, prompt=prompt, constraints=constraints)
    rel_paths = _to_relative_paths(base_url, [c.url for c in collected])

    print(f"\n[Gateway] Agent collected {len(collected)} URLs")
    if collected:
        print("[Gateway] Collected URLs:")
        for i, c in enumerate(collected, 1):
            print(f"  {i}. {c.url} - {c.title or 'No title'}")

    if not rel_paths:
        print("\n[Gateway] ERROR: Agent collected 0 URLs. Possible reasons:", file=sys.stderr)
        print("  - LLM could not find relevant documents", file=sys.stderr)
        print("  - Pages were filtered out by heuristics", file=sys.stderr)
        print("  - Network/scraping errors", file=sys.stderr)
        print("  - Try increasing --max-depth or --max-pages", file=sys.stderr)
        raise RuntimeError("agent_collected_0_urls")

    # Minimal structure: keep gateway thin; group all under one topic for the demo.
    # (Scraper service owns the heavy logic: scraping, filtering, quality gates, minio/postgres persistence.)
    config = {
        "base_url": base_url.rstrip("/"),
        "structure": {
            "demo": rel_paths,
        },
        "options": {
            "output_format": output_format,
            "include_images": bool(include_images),
            "image_handling": image_handling,
            "noise_selectors": ["header", "footer", "nav", "aside"],
            "include_metadata": True,
        },
    }

    address = f"{host}:{port}"
    async with grpc.aio.insecure_channel(address) as channel:
        stub = scraper_pb2_grpc.WebScraperServiceStub(channel)
        req = scraper_pb2.ScrapeConfiguredRequest(
            job_id=job_id,
            client_id="demo-gateway-phase3",
            config_json=json.dumps(config, ensure_ascii=False),
        )

        print(f"Connecting to {address}")
        print(f"Demo Job ID: {job_id}")
        print(f"Agent collected URLs: {len(rel_paths)}")
        print("Streaming scrape progress...\n")

        async for progress in stub.ScrapeConfigured(req):
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
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=50)

    parser.add_argument("--output-format", choices=["pdf", "docx", "both"], default="pdf")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument(
        "--image-handling",
        choices=["embed", "link_only", "both", "embed_or_link"],
        default="embed_or_link",
    )

    # LLM provider selection
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama", help="LLM provider to use")
    
    # Ollama LLM config for agent navigation
    parser.add_argument("--ollama-host", default="localhost", help="Ollama host")
    parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama port")
    parser.add_argument("--ollama-model", default="qwen2.5:3b", help="Ollama model name")
    
    # OpenAI LLM config for agent navigation
    default_openai_key = os.getenv("OPENAI_API_KEY")
    parser.add_argument("--openai-api-key", default=default_openai_key, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"), help="OpenAI base URL (for OpenAI-compatible APIs, or set OPENAI_BASE_URL env var)")
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"), help="OpenAI model name (e.g., gpt-4o, gpt-4, gpt-3.5-turbo)")

    args = parser.parse_args()
    await run_demo(
        host=args.host,
        port=args.port,
        base_url=args.base_url,
        prompt=args.prompt,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        output_format=args.output_format,
        include_images=bool(args.include_images),
        image_handling=args.image_handling,
        llm_provider=args.llm_provider,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        ollama_model=args.ollama_model,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model,
    )


if __name__ == "__main__":
    asyncio.run(main())


