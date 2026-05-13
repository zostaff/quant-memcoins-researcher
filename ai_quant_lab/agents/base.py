"""Anthropic SDK wrapper.

Centralizes:
    - retries with exponential backoff on transient errors
    - extracting the first JSON object from a response (Claude often wraps it
      in markdown fences or prose)
    - prompt caching for the system prompt when the same context is reused
      across multiple agents in a loop

Every other agent module talks to Claude exclusively through `call_claude`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from ai_quant_lab.config import settings


@dataclass(frozen=True)
class AgentMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class AgentResponse:
    text: str
    usage: dict[str, int]
    model: str
    stop_reason: str | None = None


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def extract_first_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a Claude response.

    Tries direct json.loads first, then strips markdown fences, then matches
    the first {...} block. Raises ValueError if nothing parses.
    """
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Found JSON-like block but could not parse: {exc}") from exc

    raise ValueError(f"No JSON object found in response:\n{cleaned[:500]}")


def call_claude(
    system: str,
    messages: Sequence[AgentMessage],
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.4,
    cache_system: bool = True,
    max_retries: int = 3,
) -> AgentResponse:
    """Call Claude with retries and (optionally) prompt caching on the system block.

    Args:
        system: System prompt. If `cache_system` is True, marked as cache_control
            so repeated calls hit the cache (5-minute TTL).
        messages: User/assistant turns.
        model: Override the configured model.
        max_tokens: Output budget.
        temperature: Sampling temperature. 0.4 is a good default for structured tasks.
        cache_system: Whether to apply ephemeral caching to the system block.
        max_retries: Number of retries on transient API errors.

    Returns:
        AgentResponse with the text, usage dict, and model id.

    Raises:
        RuntimeError: if no API key is configured or all retries fail.
    """
    api_key = settings.require_api_key()
    try:
        from anthropic import Anthropic  # noqa: PLC0415 — optional dep at module level
    except ImportError as exc:
        raise ImportError("Install the SDK: pip install anthropic") from exc

    client = Anthropic(api_key=api_key)
    use_model = model or settings.model

    system_blocks: list[dict[str, Any]]
    if cache_system:
        system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system_blocks = [{"type": "text", "text": system}]

    api_messages = [{"role": m.role, "content": m.content} for m in messages]

    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=use_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_blocks,
                messages=api_messages,
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            }
            return AgentResponse(
                text=text,
                usage=usage,
                model=response.model,
                stop_reason=response.stop_reason,
            )
        except Exception as exc:  # noqa: BLE001 — broad catch is intentional; we retry below
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay *= 2.0

    raise RuntimeError(f"Claude call failed after {max_retries} attempts: {last_error}")
