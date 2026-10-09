"""The wire shape of one telemetry report: what an exporter sends or stores for one window"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, TypeAlias, TypeVar

from litellm.telemetry.histogram import ATTEMPT_BOUNDS, BLOCK_COUNT_BOUNDS, LATENCY_BOUNDS_MS, Histogram
from litellm.telemetry.records import (
    AttemptRecord,
    BlockType,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    TelemetryLevel,
    TokenCounts,
    UIAction,
    UIEvent,
)

REPORT_SCHEMA_VERSION: Final = 1

JsonValue: TypeAlias = "str | int | float | bool | None | Sequence[JsonValue] | Mapping[str, JsonValue]"

_K: Final = TypeVar("_K", bound=str)


def add_counts(counts: tuple[tuple[_K, int], ...], increments: Iterable[tuple[_K, int]]) -> tuple[tuple[_K, int], ...]:
    pairs: Final = tuple(chain(counts, increments))
    keys: Final = sorted({key for key, _ in pairs})
    return tuple((key, sum(n for k, n in pairs if k == key)) for key in keys)


@dataclass(frozen=True, slots=True)
class RequestKey:
    endpoint: str
    provider: str | None
    deployment_hash: str | None
    litellm_status: StatusClass
    provider_status: StatusClass
    litellm_cache_hit: bool
    provider_cache_hit: bool
    stream: bool

    @classmethod
    def of(cls, record: RequestRecord) -> "RequestKey":
        return cls(
            endpoint=record.endpoint,
            provider=record.provider,
            deployment_hash=record.deployment_hash,
            litellm_status=record.litellm_status,
            provider_status=record.provider_status,
            litellm_cache_hit=record.litellm_cache_hit,
            provider_cache_hit=record.provider_cache_hit,
            stream=record.stream,
        )


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    request_count: int
    tokens: TokenCounts
    block_count: Histogram
    block_types: tuple[tuple[BlockType, int], ...]
    header_keys: tuple[tuple[str, int], ...]
    provider_attempts: Histogram
    latency_total: Histogram
    latency_to_headers: Histogram
    latency_to_first_token: Histogram

    @classmethod
    def of(cls, record: RequestRecord) -> "RequestMetrics":
        return cls(
            request_count=1,
            tokens=record.tokens,
            block_count=Histogram.of(BLOCK_COUNT_BOUNDS, None if record.blocks is None else record.blocks.total),
            block_types=add_counts((), () if record.blocks is None else record.blocks.by_type),
            header_keys=add_counts((), ((key, 1) for key in record.header_keys)),
            provider_attempts=Histogram.of(ATTEMPT_BOUNDS, record.provider_attempts),
            latency_total=Histogram.of(LATENCY_BOUNDS_MS, record.latency_total_ms),
            latency_to_headers=Histogram.of(LATENCY_BOUNDS_MS, record.latency_to_headers_ms),
            latency_to_first_token=Histogram.of(LATENCY_BOUNDS_MS, record.latency_to_first_token_ms),
        )

    def merge(self, other: "RequestMetrics") -> "RequestMetrics":
        return RequestMetrics(
            request_count=self.request_count + other.request_count,
            tokens=TokenCounts(
                input=self.tokens.input + other.tokens.input,
                output=self.tokens.output + other.tokens.output,
                cache_read=self.tokens.cache_read + other.tokens.cache_read,
                cache_write=self.tokens.cache_write + other.tokens.cache_write,
            ),
            block_count=self.block_count.merge(other.block_count),
            block_types=add_counts(self.block_types, other.block_types),
            header_keys=add_counts(self.header_keys, other.header_keys),
            provider_attempts=self.provider_attempts.merge(other.provider_attempts),
            latency_total=self.latency_total.merge(other.latency_total),
            latency_to_headers=self.latency_to_headers.merge(other.latency_to_headers),
            latency_to_first_token=self.latency_to_first_token.merge(other.latency_to_first_token),
        )


@dataclass(frozen=True, slots=True)
class AttemptKey:
    provider: str
    deployment_hash: str | None
    provider_status: StatusClass
    stream: bool

    @classmethod
    def of(cls, record: AttemptRecord) -> "AttemptKey":
        return cls(
            provider=record.provider,
            deployment_hash=record.deployment_hash,
            provider_status=record.provider_status,
            stream=record.stream,
        )


@dataclass(frozen=True, slots=True)
class AttemptMetrics:
    attempt_count: int
    latency: Histogram
    latency_to_first_token: Histogram

    @classmethod
    def of(cls, record: AttemptRecord) -> "AttemptMetrics":
        return cls(
            attempt_count=1,
            latency=Histogram.of(LATENCY_BOUNDS_MS, record.latency_ms),
            latency_to_first_token=Histogram.of(LATENCY_BOUNDS_MS, record.latency_to_first_token_ms),
        )

    def merge(self, other: "AttemptMetrics") -> "AttemptMetrics":
        return AttemptMetrics(
            attempt_count=self.attempt_count + other.attempt_count,
            latency=self.latency.merge(other.latency),
            latency_to_first_token=self.latency_to_first_token.merge(other.latency_to_first_token),
        )


@dataclass(frozen=True, slots=True)
class Report:
    instance: InstanceInfo
    window_start: float
    window_end: float
    requests: tuple[tuple[RequestKey, RequestMetrics], ...] = ()
    attempts: tuple[tuple[AttemptKey, AttemptMetrics], ...] = ()
    ui_events: tuple[tuple[UIEvent, int], ...] = ()
    dropped_records: int = 0
    schema_version: int = REPORT_SCHEMA_VERSION


def _enum_value(value: StatusClass | BlockType | UIAction | TelemetryLevel) -> str:
    return value.value


def _histogram_json(histogram: Histogram) -> JsonValue:
    return {"bounds": list(histogram.bounds), "counts": list(histogram.counts)}


def _request_json(key: RequestKey, metrics: RequestMetrics) -> JsonValue:
    return {
        "endpoint": key.endpoint,
        "provider": key.provider,
        "deployment_hash": key.deployment_hash,
        "litellm_status": _enum_value(key.litellm_status),
        "provider_status": _enum_value(key.provider_status),
        "litellm_cache_hit": key.litellm_cache_hit,
        "provider_cache_hit": key.provider_cache_hit,
        "stream": key.stream,
        "request_count": metrics.request_count,
        "input_tokens": metrics.tokens.input,
        "output_tokens": metrics.tokens.output,
        "cache_read_tokens": metrics.tokens.cache_read,
        "cache_write_tokens": metrics.tokens.cache_write,
        "block_count": _histogram_json(metrics.block_count),
        "block_types": {_enum_value(block_type): n for block_type, n in metrics.block_types},
        "header_keys": dict(metrics.header_keys),
        "provider_attempts": _histogram_json(metrics.provider_attempts),
        "latency_total_ms": _histogram_json(metrics.latency_total),
        "latency_to_headers_ms": _histogram_json(metrics.latency_to_headers),
        "latency_to_first_token_ms": _histogram_json(metrics.latency_to_first_token),
    }


def _attempt_json(key: AttemptKey, metrics: AttemptMetrics) -> JsonValue:
    return {
        "provider": key.provider,
        "deployment_hash": key.deployment_hash,
        "provider_status": _enum_value(key.provider_status),
        "stream": key.stream,
        "attempt_count": metrics.attempt_count,
        "latency_ms": _histogram_json(metrics.latency),
        "latency_to_first_token_ms": _histogram_json(metrics.latency_to_first_token),
    }


def _ui_event_json(event: UIEvent, count: int) -> JsonValue:
    return {"page": event.page, "action": _enum_value(event.action), "target": event.target, "count": count}


def report_to_json(report: Report) -> Mapping[str, JsonValue]:
    return {
        "schema_version": report.schema_version,
        "instance": {
            "instance_id": report.instance.instance_id,
            "litellm_version": report.instance.litellm_version,
            "telemetry_level": _enum_value(report.instance.telemetry_level),
            "config_keys": sorted(report.instance.config_keys),
        },
        "window_start": report.window_start,
        "window_end": report.window_end,
        "dropped_records": report.dropped_records,
        "requests": [_request_json(key, metrics) for key, metrics in report.requests],
        "attempts": [_attempt_json(key, metrics) for key, metrics in report.attempts],
        "ui_events": [_ui_event_json(event, count) for event, count in report.ui_events],
    }
