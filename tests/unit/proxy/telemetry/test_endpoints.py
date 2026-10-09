from dataclasses import dataclass
from collections.abc import Mapping
from typing import Final

from typing_extensions import LiteralString

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.telemetry.endpoints import router, telemetry_store
from litellm.proxy.telemetry.store import TelemetryStore


@dataclass
class _ReportTable:
    rows: tuple[Mapping[str, object], ...]
    queried_with: tuple[object, ...] | None = None

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        self.queried_with = args
        return self.rows

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        return 0


def _row(report_id: str, window_end: float) -> Mapping[str, object]:
    return {"id": report_id, "window_start": window_end - 60, "window_end": window_end, "report": {"requests": []}}


def _client(role: LitellmUserRoles, store: TelemetryStore | None) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role)
    app.dependency_overrides[telemetry_store] = lambda: store
    return TestClient(app)


def test_a_full_page_returns_a_cursor_at_its_last_report() -> None:
    table: Final = _ReportTable(rows=(_row("a", 100.0), _row("b", 100.0)))
    response: Final = _client(LitellmUserRoles.PROXY_ADMIN, TelemetryStore(table, retention_days=30)).get(
        "/telemetry/reports", params={"after": 40.0, "after_id": "z", "limit": 2}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "reports": [_row("a", 100.0), _row("b", 100.0)],
        "next_after": 100.0,
        "next_after_id": "b",
    }
    assert table.queried_with == (40.0, "z", 2)


def test_a_short_page_is_the_last_one() -> None:
    table: Final = _ReportTable(rows=(_row("a", 100.0),))
    response: Final = _client(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, TelemetryStore(table, retention_days=30)).get(
        "/telemetry/reports", params={"limit": 2}
    )
    assert response.status_code == 200, response.text
    assert (response.json()["next_after"], response.json()["next_after_id"]) == (None, None)
    assert table.queried_with == (0.0, "", 2)


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.ORG_ADMIN, LitellmUserRoles.TEAM])
def test_non_admin_roles_cannot_export_reports(role: LitellmUserRoles) -> None:
    table: Final = _ReportTable(rows=(_row("a", 100.0),))
    response: Final = _client(role, TelemetryStore(table, retention_days=30)).get("/telemetry/reports")
    assert response.status_code == 403, response.text
    assert table.queried_with is None


def test_exporting_without_a_local_store_is_an_error() -> None:
    response: Final = _client(LitellmUserRoles.PROXY_ADMIN, None).get("/telemetry/reports")
    assert response.status_code == 500, response.text
