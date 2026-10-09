import json
import uuid
from collections.abc import Awaitable
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from typing_extensions import LiteralString

from litellm._logging import verbose_proxy_logger
from litellm.telemetry.report import Report, report_to_json
from litellm.telemetry.sink import ExportOutcome

INSTANCE_ID_PARAM: Final = "telemetry_instance_id"


class Database(Protocol):
    def query_raw(self, query: LiteralString, *args: object) -> Awaitable[object]: ...
    def execute_raw(self, query: LiteralString, *args: object) -> Awaitable[int]: ...


class _InstanceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str


class StoredReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    window_start: float
    window_end: float
    report: JsonValue


_INSTANCE_ROWS: Final = TypeAdapter(tuple[_InstanceRow, ...])
_REPORT_ROWS: Final = TypeAdapter(tuple[StoredReport, ...])


class TelemetryStore:
    """The proxy DB side of telemetry: the install's stable instance id and the local report table"""

    def __init__(self, db: Database, retention_days: int) -> None:
        self._db: Final = db
        self._retention_days: Final = retention_days

    async def instance_id(self) -> str:
        await self._db.execute_raw(
            """INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES ($1, to_jsonb($2::text))
            ON CONFLICT (param_name) DO NOTHING""",
            INSTANCE_ID_PARAM,
            uuid.uuid4().hex,
        )
        rows: Final = _INSTANCE_ROWS.validate_python(
            await self._db.query_raw(
                """SELECT param_value #>> '{}' AS instance_id FROM "LiteLLM_Config" WHERE param_name = $1""",
                INSTANCE_ID_PARAM,
            )
        )
        return rows[0].instance_id

    async def save(self, report: Report) -> None:
        await self._db.execute_raw(
            """INSERT INTO "LiteLLM_TelemetryReport" (id, window_start, window_end, report)
            VALUES ($1, $2, $3, $4::jsonb)""",
            uuid.uuid4().hex,
            report.window_start,
            report.window_end,
            json.dumps(report_to_json(report)),
        )

    async def prune(self) -> None:
        await self._db.execute_raw(
            """DELETE FROM "LiteLLM_TelemetryReport" WHERE created_at < now() - make_interval(days => $1::int)""",
            self._retention_days,
        )

    async def reports_after(self, window_end: float, report_id: str, limit: int) -> tuple[StoredReport, ...]:
        return _REPORT_ROWS.validate_python(
            await self._db.query_raw(
                """SELECT id, window_start, window_end, report FROM "LiteLLM_TelemetryReport"
                WHERE (window_end, id) > ($1, $2) ORDER BY window_end, id LIMIT $3""",
                window_end,
                report_id,
                limit,
            )
        )


class LocalTableExporter:
    """``Exporter`` that keeps reports in the proxy DB for manual export instead of sending them"""

    def __init__(self, store: TelemetryStore) -> None:
        self._store: Final = store

    async def export(self, report: Report) -> ExportOutcome:
        from prisma.errors import PrismaError

        if not (report.requests or report.attempts or report.ui_events or report.dropped_records):
            return ExportOutcome.SENT
        try:
            await self._store.save(report)
        except (PrismaError, OSError) as e:
            verbose_proxy_logger.debug("telemetry: could not store the report locally: %s", e)
            return ExportOutcome.RETRY
        try:
            await self._store.prune()
        except (PrismaError, OSError) as e:
            verbose_proxy_logger.debug("telemetry: could not prune old local reports: %s", e)
        return ExportOutcome.SENT
