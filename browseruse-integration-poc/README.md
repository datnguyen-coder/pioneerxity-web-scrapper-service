# BrowserUse Integration POC (Browser Agent)

## Goal
Integrate **browser-use** (AI browser agent) into the PioneerXity platform so an end user can:
- Submit a natural-language browser task from the **client**
- The request goes through the **gateway**
- Gateway forwards it to a BrowserUse runtime service
- The user sees execution progress (logs + screenshot updates)
- The task completes with a visible final result (or a visible failure)

This is **POC scope only**.

## Target Repos
- Backend gateway: `d:/backup/Work/pioneerxity/platform/gateway`
- Frontend client: `d:/backup/Work/pioneerxity/platform/client`

## Non-Goals
- Anti-bot / stealth tuning
- CAPTCHA solving
- Multi-tenant isolation
- Concurrency/scalability hardening
- Long-term persistence of browser sessions

## Recommended POC Path
Start with **Path B (Dedicated microservice)** because:
- Gateway already has SSE streaming patterns (see `poc_cyon`)
- Client already has an SSE parsing helper (see `features/poc/cyon/streaming.ts`)
- browser-use supports step callbacks/hooks with screenshot + thought/action context

Path A (Gradio web-ui) is still described for fastest possible bring-up.

## Document Index
- `01_requirements_and_user_journey.md`
- `02_architecture_options.md`
- `03_data_contract.md`
- `04_browseruse_service_path_b.md`
- `05_gateway_integration.md`
- `06_client_ui.md`
- `07_runbook.md`

## Repo note
For this codebase, the BrowserUse microservice is implemented at:
- `browser-use-service/`

## Repo patterns this POC should follow (important)
- **Gateway streaming proxy pattern**:
  - See `gateway/src/app/modules/poc_cyon/controller.py` (streams upstream bytes when `Accept: text/event-stream`)
- **Client SSE parsing**:
  - See `client/src/features/poc/cyon/streaming.ts` (`streamSse` reads `data:` lines and JSON.parse)
- **Docker base image**:
  - Other services that need browser automation use Playwright image: `mcr.microsoft.com/playwright/python:v1.53.0-noble`.

## UX constraint
No UI animations for this POC feature (keep UI static).
