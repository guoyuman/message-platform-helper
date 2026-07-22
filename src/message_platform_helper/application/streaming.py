"""Streaming event helpers."""

from __future__ import annotations

from typing import Iterable

from ..models import HelperResponse, StreamingEvent, to_jsonable


def response_to_streaming_events(response: HelperResponse) -> Iterable[StreamingEvent]:
    sequence = 1
    yield StreamingEvent(
        type="Reasoning",
        data={"decision": to_jsonable(response.decision)},
        trace_id=response.trace_id,
        request_id=response.request_id,
        conversation_id=response.conversation_id,
        sequence=sequence,
    )
    sequence += 1
    for chunk in response.retrieved:
        yield StreamingEvent(
            type="Source",
            data={"source": to_jsonable(chunk)},
            trace_id=response.trace_id,
            request_id=response.request_id,
            conversation_id=response.conversation_id,
            sequence=sequence,
        )
        sequence += 1
    for result in response.results:
        for step in result.steps:
            yield StreamingEvent(
                type="ToolCall",
                data={"agent": result.agent, "step": to_jsonable(step)},
                trace_id=response.trace_id,
                request_id=response.request_id,
                conversation_id=response.conversation_id,
                sequence=sequence,
            )
            sequence += 1
    yield StreamingEvent(
        type="Done" if response.ok else "Error",
        data={"ok": response.ok, "issues": response.issues, "metrics": response.metrics},
        trace_id=response.trace_id,
        request_id=response.request_id,
        conversation_id=response.conversation_id,
        sequence=sequence,
    )


__all__ = ["response_to_streaming_events"]
