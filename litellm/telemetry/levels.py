import dataclasses
from typing import Final

from litellm.telemetry.records import (
    ALLOWED_HEADER_KEYS,
    OTHER_HEADER_KEY,
    AttemptRecord,
    InstanceInfo,
    RequestRecord,
    TelemetryLevel,
    UIEvent,
)
from litellm.telemetry.sink import TelemetrySink

_NO_KEYS: Final[frozenset[str]] = frozenset()
_OTHER_KEYS: Final[frozenset[str]] = frozenset({OTHER_HEADER_KEY})


def _allowlisted_header_keys(header_keys: frozenset[str]) -> frozenset[str]:
    lowered: Final = frozenset(key.lower() for key in header_keys)
    unknown: Final = lowered - ALLOWED_HEADER_KEYS
    return (lowered & ALLOWED_HEADER_KEYS) | (_OTHER_KEYS if unknown else _NO_KEYS)


class LevelGatedSink:
    """Drops every field the configured level does not allow before it reaches the wrapped sink."""

    def __init__(self, inner: TelemetrySink, level: TelemetryLevel) -> None:
        self._inner = inner
        self._level = level

    def set_instance(self, info: InstanceInfo) -> None:
        match self._level:
            case TelemetryLevel.OFF:
                return
            case TelemetryLevel.BASIC:
                self._inner.set_instance(dataclasses.replace(info, config_keys=frozenset()))
            case TelemetryLevel.FULL:
                self._inner.set_instance(info)

    def record_request(self, record: RequestRecord) -> None:
        match self._level:
            case TelemetryLevel.OFF:
                return
            case TelemetryLevel.BASIC:
                self._inner.record_request(dataclasses.replace(record, blocks=None, header_keys=frozenset()))
            case TelemetryLevel.FULL:
                self._inner.record_request(
                    dataclasses.replace(record, header_keys=_allowlisted_header_keys(record.header_keys))
                )

    def record_attempt(self, record: AttemptRecord) -> None:
        if self._level is TelemetryLevel.OFF:
            return
        self._inner.record_attempt(record)

    def record_ui_event(self, event: UIEvent) -> None:
        if self._level is not TelemetryLevel.FULL:
            return
        self._inner.record_ui_event(event)

    async def flush(self) -> None:
        if self._level is TelemetryLevel.OFF:
            return
        await self._inner.flush()
