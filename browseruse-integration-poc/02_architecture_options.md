# 02. Architecture Options

All 3 paths are acceptable for the POC. Start with A if you want a "today" demo; start with B if you want a clean platform integration.

## Path A — Gradio Web UI as Backend (Fastest)

### What
Run `browser-use/web-ui` in Docker and call its Gradio API from gateway.

### How (high-level)
- Host web-ui container (default Gradio port often `7860`).
- Gateway calls Gradio HTTP API.

### How to discover the correct Gradio endpoint (important)
Gradio apps usually expose:
- `GET /config` (or similar) which returns `dependencies` and `api_name`s.

Practical method:
1) Open the web-ui in a browser.
2) Fetch `http://<host>:7860/config`.
3) Look for the function you want (e.g. `run_agent`) in `dependencies`.
4) If `api_name` is present, prefer the named endpoint.
5) Otherwise call `/api/predict` with the correct `fn_index`.

### Pros
- Almost zero code
- Already includes agent loop + screenshots

### Cons
- Black box
- Streaming format can be awkward
- Long-running jobs and concurrency not clean

### POC recommendation
Use Path A for the need of a demo immediately. Move to Path B afterward.


## Path B — Dedicated BrowserUse Microservice (Recommended)

### What
Create a new Python microservice:
- FastAPI REST
- SSE streaming
- Runs browser-use agent per request

### Why it fits this repo
- Gateway already proxies SSE (see `poc_cyon`)
- Client already parses SSE (see `features/poc/cyon/streaming.ts`)
- Services already use `uv` and Playwright Docker base image


## Path C — MCP Server (Optional)

### What
Run browser-use as an MCP server.

### Pros
- Standard tool protocol

### Cons
- More protocol complexity
- Harder to integrate into existing web streaming flows

### POC recommendation
Do not use unless you are already building MCP-based tool routing.
