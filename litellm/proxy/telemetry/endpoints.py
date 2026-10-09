from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.telemetry.store import StoredReport, TelemetryStore

router: Final = APIRouter()


class TelemetryReportsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    reports: tuple[StoredReport, ...]
    next_after: float | None
    next_after_id: str | None


def telemetry_store() -> TelemetryStore | None:
    from litellm.proxy.proxy_server import telemetry_runtime

    return telemetry_runtime.store


@router.get("/telemetry/reports", tags=["Telemetry"], response_model=TelemetryReportsResponse)
async def export_telemetry_reports(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    store: Annotated[TelemetryStore | None, Depends(telemetry_store)],
    after: Annotated[float, Query(description="window_end of the last report already exported")] = 0.0,
    after_id: Annotated[str, Query(description="id of the last report already exported")] = "",
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> TelemetryReportsResponse:
    """Stored telemetry reports, oldest first, for installs that keep telemetry local instead of sending it.
    Page with ``after=next_after&after_id=next_after_id`` until ``next_after`` is null"""
    if user_api_key_dict.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        raise HTTPException(status_code=403, detail="Only proxy admin roles can export telemetry reports")
    if store is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    reports: Final = await store.reports_after(after, after_id, limit)
    if len(reports) < limit:
        return TelemetryReportsResponse(reports=reports, next_after=None, next_after_id=None)
    return TelemetryReportsResponse(reports=reports, next_after=reports[-1].window_end, next_after_id=reports[-1].id)
