# 07. Runbook (How to run the)

## Services involved
- Client: `client` (Next.js)
- Gateway: `gateway` (FastAPI)
- BrowserUse service (new): `browser-use-service` (FastAPI + browser-use)

All should join Docker network:
- `pioneerxity` (external) (matches existing service compose files)

## 1) Start BrowserUse service

### Required
- Playwright + Chromium
- browser-use
- LLM API key

### Environment variables
- `BROWSER_USE_INTERNAL_API_KEY=<shared_secret>`
- `OPENAI_API_KEY=<your_openai_key>` (if using OpenAI)

## 2) Configure gateway
Set in gateway `.env`:
- `BROWSER_USE_API_BASE_URL=http://browser-use-service:9010/api/v1`
- `BROWSER_USE_INTERNAL_API_KEY=<shared_secret>`

## 3) Configure client
Set in client env:
- `NEXT_PUBLIC_GATEWAY_URL=http://localhost:8000`

## 4) Run
- Start BrowserUse service
- Start gateway
- Start client

Open:
- `http://localhost:3000/en/browser-agent`

## Known limitations
- Some sites block automation.
- No CAPTCHA solving.
- High CPU/RAM use.
- Tasks can time out depending on target site and model.

## Quick health checks
- Gateway reachable: `GET /api/health` (if exists) or `GET /`.
- BrowserUse service reachable: add a simple `/health` in the service.
- Streaming works: start a run and ensure the client receives multiple `step` events.
