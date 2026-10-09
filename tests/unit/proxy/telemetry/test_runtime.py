import asyncio
from typing import Final

import pytest

from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.proxy.telemetry.runtime import TelemetryRuntime, TelemetrySettings, deployment_hasher


@pytest.mark.parametrize(
    "settings",
    [
        TelemetrySettings(),
        TelemetrySettings(level="off", endpoint="https://telemetry.example"),
        TelemetrySettings(level="verbose", endpoint="https://telemetry.example"),
        TelemetrySettings(level="basic"),
    ],
)
@pytest.mark.asyncio
async def test_telemetry_stays_off_unless_a_known_level_and_an_endpoint_are_both_set(
    settings: TelemetrySettings,
) -> None:
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(litellm_version="1.0.0", settings=settings, db=lambda: None, register=registered.append)
    assert runtime.sink is None
    assert registered == []


@pytest.mark.asyncio
async def test_an_enabled_runtime_registers_the_attempt_logger_and_stops_cleanly() -> None:
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(level="BASIC", endpoint="http://127.0.0.1:9", flush_interval_seconds=3600),
        db=lambda: None,
        register=registered.append,
    )
    assert runtime.sink is not None
    assert len(registered) == 1
    await runtime.stop()
    assert runtime.sink is None


def test_deployment_hashes_are_stable_per_install_and_differ_across_installs() -> None:
    assert deployment_hasher("install-a")("model-1") == deployment_hasher("install-a")("model-1")
    assert deployment_hasher("install-a")("model-1") != deployment_hasher("install-b")("model-1")
    assert "model-1" not in deployment_hasher("install-a")("model-1")


@pytest.mark.asyncio
async def test_stop_waits_for_in_flight_request_finalizers_before_the_last_flush() -> None:
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(level="basic", endpoint="http://127.0.0.1:9", flush_interval_seconds=3600),
        db=lambda: None,
        register=lambda _logger: None,
    )
    finished: Final[list[bool]] = []  # mutable-ok: records that the finalizer ran to completion

    async def _finalizer() -> None:
        await asyncio.sleep(0.01)
        finished.append(True)

    runtime.spawn(_finalizer())
    await runtime.stop()
    assert finished == [True]


@pytest.mark.asyncio
async def test_stopping_a_runtime_that_never_started_is_a_no_op() -> None:
    runtime: Final = TelemetryRuntime()
    await runtime.stop()
    assert runtime.sink is None
