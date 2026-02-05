# 04. BrowserUse Service (Path B) — Implementation Spec

## Service responsibilities
- Accept a browser task request from gateway.
- Run browser-use agent.
- Stream step events (logs + screenshot snapshots) via SSE.
- Return final result or error.

Gateway will:
- Authenticate end user
- Forward request + internal auth header

## Service shape
Create a new microservice folder at repo root:
- `platform/browser-use-service/`

Follow other services:
- Use `uv` (`pyproject.toml`)
- Use Docker base image that already exists in repo for browser automation:
  - `mcr.microsoft.com/playwright/python:v1.53.0-noble`

## Environment variables
- `BROWSER_USE_INTERNAL_API_KEY` (shared secret expected in `X-Gateway-Key` header)
- LLM provider vars depending on what you use:
  - OpenAI: `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`)
- Optional tuning:
  - `BROWSER_USE_SETUP_LOGGING=false` if you want to reduce noisy logging

## HTTP API

### POST /api/v1/browser-use/run/stream
- Auth: requires header `X-Gateway-Key: <shared-secret>`
- Body:

```json
{
  "task": "Find a flight to Tokyo under $500",
  "maxSteps": 20,
  "model": "gpt-5-mini",
  "headless": true
}
```

- Response: `text/event-stream` with events defined in `03_data_contract.md`


## How to run browser-use and capture steps
browser-use supports:
- `register_new_step_callback(state: BrowserStateSummary, model_output: AgentOutput, steps: int)`
- `await agent.run(on_step_start=..., on_step_end=...)`

POC recommendation:
- Use `register_new_step_callback` because it gives you:
  - `state.url`, `state.title`, `state.screenshot` (base64)
  - `model_output.current_state.*` and `model_output.action`

### Step callback → SSE event mapping
For each step callback:
- `step`: `steps`
- `action`: first action in `model_output.action` (use its keys)
- `agentThought`: build from:
  - `model_output.current_state.evaluation_previous_goal`
  - `model_output.current_state.next_goal`
  - `model_output.current_state.memory`
- `screenshot`: `data:image/png;base64,<state.screenshot>` if present

### Final result
After `history = await agent.run(...)`:
- `finalText`: `history.final_result()`
- `success`: `history.is_successful()` (or `history.is_done()` + no errors)


## Implementation notes (SSE)
- Use a per-request async generator that yields serialized SSE chunks.
- Emit a terminal `[DONE]` line.

### Example SSE send order
1) `log` event: "starting"
2) N x `step` events
3) `result` or `error`
4) `[DONE]`

## Dockerization

### Dockerfile
Use same pattern as gateway/agent-specialist:
- install `uv`
- `uv venv` + `uv sync`
- run with `python -m ...`

Base image must include Playwright browsers:
- `mcr.microsoft.com/playwright/python:v1.53.0-noble`

### docker-compose.yml
Match other services:
- join external network `pioneerxity`
- expose port (example `9010:9010`)
- `env_file: .env`

## Known limitations (POC)
- Memory usage can be high due to Chromium.
- No concurrency guarantees.
- Some sites will block automation.
- No CAPTCHA solving.
