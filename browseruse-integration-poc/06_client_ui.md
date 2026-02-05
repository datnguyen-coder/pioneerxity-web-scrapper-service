# 06. Client UI (Browser Agent Page)

## Where to place the UI
Follow the existing POC structure:
- Cyon lives under `client/src/app/[lang]/poc/cyon/...`

Add a new POC route:
- `client/src/app/[lang]/browser-agent/page.tsx`

And add it to sidebar POC nav:
- `client/src/components/app-sidebar.tsx`

## Routing
Sidebar currently builds `pocItems` with one item (Cyon). Add another entry:
- Name: `BrowserAgent`
- URL: `/${locale}/browser-agent`

## Data fetching & streaming
Create a new feature folder:
- `client/src/features/browser-agent/`

Files:
- `api.ts` (types + request payload)
- `streaming.ts` (SSE helper call)

Use the existing SSE parsing helper style (copy pattern from `features/poc/cyon/streaming.ts`).

### Client → gateway URL
Gateway base is in `client/src/lib/config.ts` as `gatewayApiBaseUrl`.
Create:
- `export const browserAgentApiBaseUrl = `${gatewayApiBaseUrl}/browser-agent``

Streaming call should hit:
- `POST ${browserAgentApiBaseUrl}/run/stream`

## UI components
Minimal POC UI (no animations):

### Inputs
- Textarea: task instruction
- Optional inputs:
  - maxSteps
  - model

### Controls
- Start button
- Stop button (AbortController)

### Output
- Latest screenshot preview (img)
- Log list:
  - each step shows step number + action + description
  - show url/title if available
- Final result:
  - show finalText
- Error:
  - show error message clearly

## No animation rule
Do not add new transition/hover animation classes for this feature.

## Minimal “done” checklist
- User can open `/en/browser-agent`.
- User can start a run.
- User sees step logs.
- User sees screenshot updates.
- User sees final result.
