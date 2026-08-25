"""AG-UI event normalization helpers."""

import json
from collections.abc import AsyncIterator
from typing import Any


def to_sse(event: dict[str, Any]) -> str:
    """Serialize an event for a Server-Sent Events response."""

    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def as_sse(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for event in events:
        yield to_sse(event)
