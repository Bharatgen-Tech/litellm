import asyncio
import contextlib
import hashlib
import uuid
from collections.abc import Callable, Coroutine
from typing import Final

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.telemetry.aggregate import AggregatingSink
from litellm.telemetry.http_exporter import HttpExporter
from litellm.telemetry.levels import LevelGatedSink
from litellm.telemetry.records import InstanceInfo, TelemetryLevel
from litellm.telemetry.sink import TelemetrySink


class TelemetrySettings(BaseSettings):
    """``LITELLM_TELEMETRY_*`` env vars"""

    model_config = SettingsConfigDict(
        env_prefix="LITELLM_TELEMETRY_", case_sensitive=False, extra="ignore", frozen=True
    )

    level: str = TelemetryLevel.OFF.value
    endpoint: str | None = None
    flush_interval_seconds: float = 60.0
    settle_timeout_seconds: float = 2.0


def deployment_hasher(salt: str) -> Callable[[str], str]:
    return lambda model_id: hashlib.sha256(f"{salt}:{model_id}".encode()).hexdigest()[:16]


class TelemetryRuntime:
    """Owns the proxy's telemetry sink, its flush loop and the finalizer tasks the middleware spawns"""

    def __init__(self) -> None:
        self.sink: TelemetrySink | None = None
        self._client: httpx.AsyncClient | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._pending: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs keep finalizers alive

    def spawn(self, coroutine: Coroutine[None, None, None]) -> None:
        task: Final = asyncio.create_task(coroutine)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def start(
        self, *, litellm_version: str, settings: TelemetrySettings, register: Callable[[TelemetryAttemptLogger], None]
    ) -> None:
        level: Final = TelemetryLevel.parse(settings.level)
        if level is None:
            verbose_proxy_logger.warning(
                "telemetry: unknown LITELLM_TELEMETRY_LEVEL %r, leaving it off", settings.level
            )
            return
        if level is TelemetryLevel.OFF:
            return
        if settings.endpoint is None:
            verbose_proxy_logger.warning(
                "telemetry: LITELLM_TELEMETRY_LEVEL is set but LITELLM_TELEMETRY_ENDPOINT is not"
            )
            return
        instance: Final = InstanceInfo(
            instance_id=uuid.uuid4().hex, litellm_version=litellm_version, telemetry_level=level
        )
        client: Final = httpx.AsyncClient(timeout=10.0)
        sink: Final = LevelGatedSink(AggregatingSink(HttpExporter(client, settings.endpoint)), level)
        sink.set_instance(instance)
        register(TelemetryAttemptLogger(sink, deployment_hasher(instance.instance_id)))
        self.sink, self._client = sink, client
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
        if self._client is not None:
            await self._client.aclose()
        self.sink = None
