# 05. Gateway Integration

## Why gateway should proxy
- Gateway already owns auth.
- Client already talks to gateway base URL.
- This matches the existing Cyon PoC proxy pattern.

## Add gateway config
In `gateway/src/configs/configuration.py`, add settings similar to Cyon:
- `BROWSER_USE_API_BASE_URL` (default `http://localhost:9010/api/v1`)
- `BROWSER_USE_INTERNAL_API_KEY` (optional)
- `BROWSER_USE_PROXY_TIMEOUT` (default 60)

## Add a new module
Create:
- `gateway/src/app/modules/browser_use/controller.py`

Router prefix:
- `/api/browser-use`

## API endpoints (gateway)

### POST /api/browser-use/run/stream
- Requires authenticated user (`require_authenticated_user`).
- Proxies to BrowserUse service `POST {BROWSER_USE_API_BASE_URL}/browser-use/run/stream`.
- Adds internal header:
  - `X-Gateway-Key: <BROWSER_USE_INTERNAL_API_KEY>`
- Sets request `Accept: text/event-stream` to ensure upstream streams.

Streaming proxy behavior:
- Copy the exact streaming proxy approach from:
  - `gateway/src/app/modules/poc_cyon/controller.py`

Key details to keep:
- Detect streaming by `Accept: text/event-stream`.
- Use `httpx.AsyncClient(...).stream(...)`.
- Return `StreamingResponse(upstream.aiter_bytes(), ...)`.
- Remove hop-by-hop headers.
- Add `X-Accel-Buffering: no`.

## Auth behavior
- End-user auth: gateway JWT (already implemented).
- Internal service auth: `X-Gateway-Key` only.
- Do not forward the end-user JWT to the microservice.

## Error handling
- If upstream is unreachable: return 502 with helpful message.
- If upstream returns non-200: forward status and body.

## Minimal “done” checklist
- Gateway has `/api/browser-use/run/stream`.
- Authenticated users can access it.
- Streaming works through gateway.
