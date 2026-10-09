import asyncio
import contextlib
import hashlib
import uuid
from collections.abc import Callable, Coroutine
from typing import Final

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # legacy params: dict signature
)
from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.proxy.telemetry.store import Database, LocalTableExporter, TelemetryStore
from litellm.telemetry.aggregate import AggregatingSink
from litellm.telemetry.http_exporter import HttpExporter
from litellm.telemetry.levels import LevelGatedSink
from litellm.telemetry.records import InstanceInfo, TelemetryLevel
from litellm.telemetry.sink import Exporter, TelemetrySink
from litellm.types.llms.custom_http import httpxSpecialProvider


class TelemetrySettings(BaseSettings):
    """``LITELLM_TELEMETRY_*`` env vars"""

    model_config = SettingsConfigDict(
        env_prefix="LITELLM_TELEMETRY_", case_sensitive=False, extra="ignore", frozen=True
    )

    level: str = TelemetryLevel.OFF.value
    endpoint: str | None = None
    flush_interval_seconds: float = 60.0
    settle_timeout_seconds: float = 2.0
    retention_days: int = 30


def deployment_hasher(salt: str) -> Callable[[str], str]:
    return lambda model_id: hashlib.sha256(f"{salt}:{model_id}".encode()).hexdigest()[:16]


def _shared_client() -> httpx.AsyncClient:
    return get_async_httpx_client(httpxSpecialProvider.LoggingCallback, params={"timeout": 10.0}).client


def _exporter(endpoint: str | None, store: TelemetryStore | None) -> Exporter:
    if endpoint is not None:
        return HttpExporter(_shared_client(), endpoint)
    assert store is not None, "start() returns early when there is neither an endpoint nor a database"
    return LocalTableExporter(store)


class TelemetryRuntime:
    """Owns the proxy's telemetry sink, its flush loop and the finalizer tasks the middleware spawns"""

    def __init__(self) -> None:
        self.sink: TelemetrySink | None = None
        self.store: TelemetryStore | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._pending: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs keep finalizers alive

    def spawn(self, coroutine: Coroutine[None, None, None]) -> None:
        task: Final = asyncio.create_task(coroutine)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def start(
        self,
        *,
        litellm_version: str,
        settings: TelemetrySettings,
        db: Database | None,
        register: Callable[[TelemetryAttemptLogger], None],
    ) -> None:
        level: Final = TelemetryLevel.parse(settings.level)
        if level is None:
            verbose_proxy_logger.warning(
                "telemetry: unknown LITELLM_TELEMETRY_LEVEL %r, leaving it off", settings.level
            )
            return
        if level is TelemetryLevel.OFF:
            return
        store: Final = TelemetryStore(db, settings.retention_days) if db is not None else None
        if settings.endpoint is None and store is None:
            verbose_proxy_logger.warning(
                "telemetry: LITELLM_TELEMETRY_LEVEL is set but there is no LITELLM_TELEMETRY_ENDPOINT or database"
            )
            return
        instance: Final = InstanceInfo(
            instance_id=await store.instance_id() if store is not None else uuid.uuid4().hex,
            litellm_version=litellm_version,
            telemetry_level=level,
        )
        sink: Final = LevelGatedSink(AggregatingSink(_exporter(settings.endpoint, store)), level)
        sink.set_instance(instance)
        register(TelemetryAttemptLogger(sink, deployment_hasher(instance.instance_id)))
        self.sink, self.store = sink, store
        self._flush_task = asyncio.create_task(self._flush_every(sink, settings.flush_interval_seconds))

    @staticmethod
    async def _flush_every(sink: TelemetrySink, interval_s: float) -> None:
        while True:
            await asyncio.sleep(interval_s)
            await sink.flush()

    async def stop(self) -> None:
        sink: Final = self.sink
        if sink is None:
            return
        flush_task: Final = self._flush_task
        if flush_task is not None:
            flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await flush_task
        await asyncio.gather(*self._pending, return_exceptions=True)
        await sink.flush()
        self.sink = None
        self.store = None
