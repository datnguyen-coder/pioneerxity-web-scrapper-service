"""BrowserUse agent runner + SSE event mapping."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Optional

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList, AgentOutput
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary

from .config import RunRequest, Settings
from .llm import build_llm


def _serialize_event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _step_event(
    *,
    step: int,
    state: BrowserStateSummary,
    output: AgentOutput,
) -> dict[str, Any]:
    action = None
    description = ""
    if output and output.action:
        try:
            first = output.action[0].model_dump(exclude_none=True)
            action = list(first.keys())[0] if first else None
            description = json.dumps(first, ensure_ascii=False)
        except Exception:
            action = None
    thought = ""
    if output and output.current_state:
        cs = output.current_state
        parts = [
            getattr(cs, "evaluation_previous_goal", ""),
            getattr(cs, "next_goal", ""),
            getattr(cs, "memory", ""),
        ]
        thought = " ".join([p for p in parts if p])
    screenshot = None
    if getattr(state, "screenshot", None):
        screenshot = f"data:image/png;base64,{state.screenshot}"
    return {
        "type": "step",
        "step": step,
        "action": action or "",
        "description": description,
        "agentThought": thought,
        "url": getattr(state, "url", "") or "",
        "title": getattr(state, "title", "") or "",
        "screenshot": screenshot,
        "isComplete": False,
    }


def _result_event(history: AgentHistoryList) -> dict[str, Any]:
    final_text = history.final_result() if history else ""
    errors = history.errors() if history else []
    success = bool(final_text) and not any(errors or [])
    return {
        "type": "result",
        "success": success,
        "finalText": final_text or "",
        "errors": errors or [],
        "isComplete": True,
    }


def _error_event(error: Exception) -> dict[str, Any]:
    return {
        "type": "error",
        "error": str(error),
        "isComplete": True,
    }


async def run_agent_stream(
    *,
    req: RunRequest,
    settings: Settings,
) -> AsyncIterator[bytes]:
    queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()

    llm = build_llm(
        provider=req.provider,
        model=req.model,
        temperature=req.temperature,
        settings=settings,
    )

    browser_profile = BrowserProfile(
        headless=req.headless,
        window_size={"width": 1280, "height": 900},
    )
    browser_session = BrowserSession(browser_profile=browser_profile)

    agent = Agent(
        task=req.task,
        llm=llm,
        browser_session=browser_session,
        use_vision=True,
    )

    async def on_step(state: BrowserStateSummary, output: AgentOutput, step: int) -> None:
        await queue.put(_step_event(step=step, state=state, output=output))

    async def _maybe_await(result: object) -> None:
        if asyncio.iscoroutine(result):
            await result

    async def runner() -> None:
        try:
            agent.register_new_step_callback = on_step
            history = await agent.run(max_steps=req.maxSteps)
            await queue.put(_result_event(history))
        except Exception as e:
            await queue.put(_error_event(e))
        finally:
            await queue.put(None)
            try:
                await _maybe_await(agent.close())
            except Exception:
                pass
            try:
                await _maybe_await(browser_session.stop())
            except Exception:
                pass

    asyncio.create_task(runner())

    yield _serialize_event({"type": "log", "level": "info", "message": "starting_agent"})
    while True:
        item = await queue.get()
        if item is None:
            break
        yield _serialize_event(item)
    yield b"data: [DONE]\n\n"


