# 03. Data Contract (Gateway ↔ Client)

## Why this matters
Client already has an SSE parser that expects:
- Lines that start with `data:`
- Each `data:` payload is either:
  - JSON string for one event, or
  - `[DONE]` sentinel to stop

Reference:
- `client/src/features/poc/cyon/streaming.ts` (`streamSse`)

So for BrowserUse POC, **use the same contract**.

## SSE response (Example, not forced to follow)
- `Content-Type: text/event-stream`
- Send a stream of lines:

```
data: {"type":"step",...}

data: {"type":"step",...}

data: {"type":"result",...}

data: [DONE]

```

## Event types (Example, not forced to follow)

### 1) step
Emitted after each agent step.

```json
{
  "type": "step",
  "step": 5,
  "action": "click_element",
  "description": "Clicking 'Search Flights'",
  "agentThought": "This button leads to the results page.",
  "url": "https://example.com",
  "title": "Flights - Example",
  "screenshot": "data:image/png;base64,...",
  "isComplete": false
}
```

Notes:
- `action` comes from the browser-use `AgentOutput.action` list.
- `agentThought` can be built from `model_output.current_state.*` (evaluation/memory/next_goal).
- `screenshot` should be a **data URL** so client can render it directly.

### 2) log
Optional; can be used for simple text logs that are not tied to a step.

```json
{
  "type": "log",
  "level": "info",
  "message": "Starting agent",
  "timestamp": "2026-01-30T12:00:00Z"
}
```

### 3) result
Final result when done.

```json
{
  "type": "result",
  "success": true,
  "finalText": "Found 3 flights under $500...",
  "isComplete": true
}
```

### 4) error
Terminal error. Client should stop streaming.

```json
{
  "type": "error",
  "error": "Timeout navigating to ...",
  "isComplete": true
}
```

## Minimal client rendering requirements
- Show an ordered list of `step` events.
- Show `latestScreenshot` from the most recent `step` that has a screenshot.
- Show `finalText` on result.
- Show error clearly on error.

## Compatibility constraints
- Keep payload as JSON object.
- Do not rely on SSE `event:` fields; the current client helper ignores it.
- Always end with `[DONE]`.
