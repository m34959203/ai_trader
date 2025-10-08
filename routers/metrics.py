"""Monitoring endpoints exposing Prometheus metrics and SLO reports."""
from __future__ import annotations

from fastapi import APIRouter, Response

from monitoring.observability import OBSERVABILITY

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    data = OBSERVABILITY.export_prometheus()
    return Response(content=data, media_type=OBSERVABILITY.prometheus_content_type)


@router.get("/observability/slo")
async def slo_status() -> dict:
    return {"slos": OBSERVABILITY.slo_report()}
