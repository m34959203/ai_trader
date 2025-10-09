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


@router.get("/observability/broker-latency")
async def broker_latency_snapshot() -> dict:
    return OBSERVABILITY.broker_latency_snapshot()


@router.get("/observability/thread-health")
async def thread_health_snapshot() -> dict:
    return OBSERVABILITY.thread_failure_snapshot()


@router.get("/observability/market-data")
async def market_data_snapshot() -> dict:
    return OBSERVABILITY.market_data_snapshot()
