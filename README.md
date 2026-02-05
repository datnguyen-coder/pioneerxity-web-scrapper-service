# Pioneerxity Web Scraper Service

Internal **gRPC microservice** for configured documentation scraping (Phase 1), with progress streaming, noise filtering, document generation (PDF/DOCX), and storage (MinIO + PostgreSQL metadata).

## Who calls this service?

- **Service client**: an internal **backend gateway** (or other internal backend) that prepares the full payload and calls this gRPC service.
- **End user**: triggers workflows via a platform UI (e.g., CYON UI) through the gateway. End users do **not** call this service directly, and do not send raw parameters here.
- **Presets/config ownership**: scraping presets (document lists / discovery options / scraping options) are owned and managed by the **gateway**. This service is intentionally **preset-agnostic**: it executes the payload it receives and persists per-job configuration/metadata for traceability.

> This repository is implemented to follow the mandatory rules in:
> - `1. High-Level Design & Base Requirements.md`
> - `2. Requirements Specification.md`
> - `3. Developer Guidelines.md`
> - `4. Implementation Guidelines.md`

## Quick Start (Docker - recommended)

Starts **Postgres + MinIO + gRPC service**:

```bash
docker compose up -d
```

Notes:
- gRPC listens on `${GRPC_PORT:-50051}` (default **50051**).
- MinIO API: **9000**, MinIO Console: **9001**.
- Phase 2 also needs **Ollama** running (see below). `docker-compose.yml` does not start Ollama.

## Quick Start (Local Dev)

### 1) Install deps

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
cp env.example .env  # Windows: copy env.example .env
uv pip install -e .
python scripts/generate_proto.py
```

### Install Playwright browsers (required for local runs)

```bash
python -m playwright install chromium
```

### Install dev deps (tests)

```bash
# Using uv
uv pip install -e ".[dev]"
```

### 2) Start dependencies (Postgres + MinIO)

```bash
docker compose up -d postgres minio
```

### 3) Run service

```bash
python -m scraper.main
```

### 4) Test gRPC streaming

```bash
python test_client.py --mode configured
```

## Phase 2 (Autonomous Discovery) - Local Test

Prereqs:
- Service running
- Ollama running on `OLLAMA_HOST:OLLAMA_PORT` with model `qwen2.5:3b` (default per MD)

```bash
ollama pull qwen2.5:3b
```

```bash
python test_client_phase2.py --base-url https://www.cyon.ch/support --max-depth 2 --max-pages 30
```

## Phase 3 (Demo) - Gateway + Agent Stub (Minimal)

This is a **minimal demo** of the Phase 3 flow using:
- a **thin gateway runner** (script) and
- a **Playwright-based agent** that returns a URL list (BrowserUse-style navigation),
then calls the scraper service with Phase 1 `ScrapeConfigured`.

Prereqs:
- Service running on gRPC `localhost:50051`
- Playwright browser installed (`python -m playwright install chromium`)

Run:

```bash
python scripts/demo_phase3_gateway.py ^
  --base-url https://www.cyon.ch/support ^
  --prompt "collect all support documents" ^
  --max-depth 2 ^
  --max-pages 80
```

## Phase 3 (Service) - gRPC Integration

Phase 3 is now integrated as a gRPC method:

```
rpc ScrapePhase3(Phase3ScrapeRequest) returns (stream ScrapeProgress);
```

After editing `proto/scraper.proto`, regenerate gRPC code:

```bash
uv run python scripts/generate_proto.py
```

### Phase 2 workflow (handover)

Phase 2 uses **three RPCs** (called by the gateway/service client):
- `DiscoverStructure`: crawl deterministically within scope and ask LLM to suggest **noise selectors** (header/footer/nav/search/ads...). LLM **does not** decide topics/pages.
- `GetDiscoveredStructure`: fetch discovered structure + **merged scraping options** (including LLM-suggested `noise_selectors`).
- `ApproveAndScrape`: optional override of structure, then runs Phase 1 pipeline over **all discovered pages**.

## Notes

- The service is internal-only. Auth is intentionally not enforced (per current requirements).
- All secrets must come from environment variables. Do not commit `.env`.
- To run unit tests you need pytest installed (see "Install dev deps"). Then run: `python -m pytest -q`

### Troubleshooting

- **gRPC port already in use (50051)**: stop the old process or change `GRPC_PORT` in `.env`.
- **Phase 2 + Docker**: if the scraper runs in Docker and Ollama runs on host, set `OLLAMA_HOST=host.docker.internal`.
- **Windows PDF differences**: WeasyPrint may not be available due to native deps; service falls back to **ReportLab** automatically.


