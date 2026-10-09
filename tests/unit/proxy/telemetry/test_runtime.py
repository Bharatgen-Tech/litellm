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
    runtime.start(litellm_version="1.0.0", settings=settings, register=registered.append)
    assert runtime.sink is None
    assert registered == []


@pytest.mark.asyncio
async def test_an_enabled_runtime_registers_the_attempt_logger_and_stops_cleanly() -> None:
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(level="BASIC", endpoint="http://127.0.0.1:9", flush_interval_seconds=3600),
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
