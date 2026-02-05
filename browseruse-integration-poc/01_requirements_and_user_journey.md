# 01. Requirements & User Journey

## Functional requirements

### End-user flow (client)
- User can enter a natural-language instruction.
  - Example: "Find a flight to Tokyo under $500".
- User can start the agent.
- User can choose LLM/provider settings (e.g., OpenAI model selection, API key slot if allowed, max steps) before starting.
- User can see execution feedback while the agent runs.
  - Preferred: live or semi-live streaming.
- User can see results:
  - Final extracted content (text)
  - Screenshot updates (recommended)

### Platform flow (end-to-end)
1) Client submits instruction
2) Gateway authenticates the user and receives the instruction
3) Gateway forwards to a BrowserUse runtime service
4) BrowserUse runs the agent
5) Execution steps are streamed back (SSE)
6) Client renders step logs + screenshot snapshots
7) Task completes (success/fail) visibly

## Non-goals (explicit)
- Anti-bot / stealth tuning
- CAPTCHA solving
- Long-lived browser sessions
- Multi-tenant isolation

## POC acceptance criteria
- A logged-in user can type a browser task.
- The agent runs.
- The user sees progress (at least text; screenshots preferred).
- The task completes or fails visibly.

## UX requirements
- UI should show:
  - Input textarea
  - Start / Stop buttons
  - Streaming log list
  - Latest screenshot preview
  - Final result section

## Security assumptions (POC)
- Gateway handles end-user authentication.
- BrowserUse service can assume trusted internal calls from gateway.
- Use an internal shared secret header (`X-Gateway-Key`) to prevent direct public access.
