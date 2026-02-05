#!/usr/bin/env python3
"""Test client for Web Scraper Service (Phase 1 configured scraping)."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

import grpc

try:
    from proto.generated import scraper_pb2, scraper_pb2_grpc
except ImportError:
    print("ERROR: gRPC code not generated. Run: python scripts/generate_proto.py")
    sys.exit(1)


def sample_config() -> dict:
    return {
        # NOTE: replace with a real docs base_url + structure before running E2E.
        "base_url": "https://www.cyon.ch",
        "structure": {
            "support": {
                "e-mail": [
                    "/support/a/grosse-eines-e-mail-kontos-festlegen-beschranken"
                    ],
            },
        },
        "options": {
            # Keep E2E stable: start with images off, then enable later.
            "include_images": True,
            # Upload images + embed them in the generated PDF (Phase 1 requirement).
            "image_handling": "both",
            "output_format": "pdf",
            "noise_selectors": ["header", "footer", "nav"],
            "include_metadata": True,
        },
    }


async def configured_stream(host: str, port: int, *, config: dict):
    address = f"{host}:{port}"
    async with grpc.aio.insecure_channel(address) as channel:
        stub = scraper_pb2_grpc.WebScraperServiceStub(channel)

        req = scraper_pb2.ScrapeConfiguredRequest(
            job_id=str(uuid.uuid4()),
            client_id="test-client",
            config_json=json.dumps(config),
        )

        print(f"Connecting to {address}")
        print(f"Job ID: {req.job_id}")
        print("Streaming progress...\n")

        last_minio_path: str | None = None
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
                last_minio_path = progress.last_document.minio_path
            if progress.is_complete:
                if progress.stats:
                    print(
                        f"\nDone. total={progress.stats.total_pages_processed} "
                        f"ok={progress.stats.successful_pages} failed={progress.stats.failed_pages} "
                        f"ms={progress.stats.total_processing_time_ms}"
                    )
                break

        return last_minio_path


def _download_from_minio(minio_path: str, out_dir: str) -> None:
    """Best-effort download for local verification (test client only)."""
    try:
        from minio import Minio
        from scraper.config.settings import get_settings
    except Exception as e:
        print(f"WARNING: cannot import MinIO client/settings for download: {e}")
        return

    settings = get_settings()
    bucket = settings.minio_bucket

    # minio_path is typically: "<bucket>/<object_name>"
    object_name = minio_path
    if "/" in minio_path:
        maybe_bucket, rest = minio_path.split("/", 1)
        if maybe_bucket == bucket:
            object_name = rest

    client = Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=bool(settings.minio_secure),
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _save(obj_name: str) -> None:
        resp = client.get_object(bucket, obj_name)
        try:
            data = resp.read()
        finally:
            resp.close()
            resp.release_conn()
        dest = out / Path(obj_name).name
        dest.write_bytes(data)
        print(f"Downloaded: {dest}")

    try:
        _save(object_name)
    except Exception as e:
        print(f"WARNING: failed to download PDF from MinIO: {e}")

    # Also try to download metadata json next to the document
    try:
        _save(object_name + ".metadata.json")
    except Exception:
        pass


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--mode", choices=["configured"], default="configured")
    parser.add_argument("--base-url", default=None, help="Base URL (e.g. https://www.cyon.ch)")
    parser.add_argument(
        "--path",
        action="append",
        default=None,
        help="Relative path under base_url to scrape (repeatable). Example: /support/a/xyz",
    )
    parser.add_argument("--output-format", choices=["pdf", "docx", "both"], default=None)
    parser.add_argument("--include-images", action="store_true", help="Enable image handling")
    parser.add_argument(
        "--image-handling",
        choices=["embed", "link_only", "both", "embed_or_link"],
        default=None,
    )
    parser.add_argument("--download-dir", default=None, help="If set, download the generated PDF + metadata from MinIO")
    args = parser.parse_args()

    cfg = sample_config()

    if args.base_url:
        cfg["base_url"] = args.base_url.rstrip("/")

    if args.path:
        cfg["structure"] = {"cli": {"paths": args.path}}

    if args.output_format:
        cfg["options"]["output_format"] = args.output_format

    if args.include_images:
        cfg["options"]["include_images"] = True

    if args.image_handling:
        cfg["options"]["image_handling"] = args.image_handling

    minio_path = await configured_stream(args.host, args.port, config=cfg)
    if args.download_dir and minio_path:
        _download_from_minio(minio_path, args.download_dir)


if __name__ == "__main__":
    asyncio.run(main())


