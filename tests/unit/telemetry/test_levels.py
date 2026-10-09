from dataclasses import dataclass
from typing import Final

import pytest

from litellm.telemetry.levels import LevelGatedSink, TelemetryLevel
from litellm.telemetry.records import (
    AttemptRecord,
    BlockCounts,
    BlockType,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    UIAction,
    UIEvent,
)


@dataclass
class _RecordingSink:
    instances: tuple[InstanceInfo, ...] = ()
    requests: tuple[RequestRecord, ...] = ()
    attempts: tuple[AttemptRecord, ...] = ()
    ui_events: tuple[UIEvent, ...] = ()
    flushes: int = 0

    def set_instance(self, info: InstanceInfo) -> None:
        self.instances = (*self.instances, info)

    def record_request(self, record: RequestRecord) -> None:
        self.requests = (*self.requests, record)

    def record_attempt(self, record: AttemptRecord) -> None:
        self.attempts = (*self.attempts, record)

    def record_ui_event(self, event: UIEvent) -> None:
        self.ui_events = (*self.ui_events, event)

    async def flush(self) -> None:
        self.flushes += 1


_REQUEST: Final = RequestRecord(
    endpoint="/chat/completions",
    stream=True,
    litellm_status=StatusClass.SUCCESS,
    latency_total_ms=120.0,
    blocks=BlockCounts(total=2, by_type=((BlockType.TEXT, 1), (BlockType.IMAGE, 1))),
    header_keys=frozenset({"Anthropic-Beta", "authorization", "x-customer-secret"}),
)
_ATTEMPT: Final = AttemptRecord(provider="openai", provider_status=StatusClass.SUCCESS, stream=True, latency_ms=100.0)
_INSTANCE: Final = InstanceInfo(
    instance_id="abc", litellm_version="1.0.0", telemetry_level=TelemetryLevel.FULL, config_keys=frozenset({"cache"})
)
_UI_EVENT: Final = UIEvent(page="models", action=UIAction.VIEW)


async def _send_everything(sink: LevelGatedSink) -> None:
    sink.set_instance(_INSTANCE)
    sink.record_request(_REQUEST)
    sink.record_attempt(_ATTEMPT)
    sink.record_ui_event(_UI_EVENT)
    await sink.flush()


@pytest.mark.asyncio
async def test_off_forwards_nothing() -> None:
    inner: Final = _RecordingSink()
    await _send_everything(LevelGatedSink(inner, TelemetryLevel.OFF))
    assert inner == _RecordingSink()


@pytest.mark.asyncio
async def test_basic_strips_request_structure_config_keys_and_ui_events() -> None:
    inner: Final = _RecordingSink()
    await _send_everything(LevelGatedSink(inner, TelemetryLevel.BASIC))
    assert inner == _RecordingSink(
        instances=(InstanceInfo(instance_id="abc", litellm_version="1.0.0", telemetry_level=TelemetryLevel.FULL),),
        requests=(
            RequestRecord(
                endpoint="/chat/completions", stream=True, litellm_status=StatusClass.SUCCESS, latency_total_ms=120.0
            ),
        ),
        attempts=(_ATTEMPT,),
        flushes=1,
    )


@pytest.mark.asyncio
async def test_full_keeps_everything_but_collapses_unlisted_header_keys_into_other() -> None:
    inner: Final = _RecordingSink()
    await _send_everything(LevelGatedSink(inner, TelemetryLevel.FULL))
    assert inner.requests[0].header_keys == frozenset({"anthropic-beta", "other"})
    assert inner.requests[0].blocks == _REQUEST.blocks
    assert inner.instances == (_INSTANCE,)
    assert inner.attempts == (_ATTEMPT,)
    assert inner.ui_events == (_UI_EVENT,)
    assert inner.flushes == 1


def test_full_with_only_allowlisted_header_keys_adds_no_other_key() -> None:
    inner: Final = _RecordingSink()
    LevelGatedSink(inner, TelemetryLevel.FULL).record_request(
        RequestRecord(
            endpoint="/v1/messages",
            stream=False,
            litellm_status=StatusClass.SUCCESS,
            latency_total_ms=1.0,
            header_keys=frozenset({"anthropic-version"}),
        )
    )
    assert inner.requests[0].header_keys == frozenset({"anthropic-version"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, TelemetryLevel.OFF),
        ("", TelemetryLevel.OFF),
        ("off", TelemetryLevel.OFF),
        (" Basic ", TelemetryLevel.BASIC),
        ("FULL", TelemetryLevel.FULL),
        ("true", None),
    ],
)
def test_parse_level(raw: str | None, expected: TelemetryLevel | None) -> None:
    assert TelemetryLevel.parse(raw) is expected
