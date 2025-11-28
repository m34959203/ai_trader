"""Monitoring API endpoints for dashboard.

Provides:
- Health checks
- Metrics data
- System status
- Real-time updates
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
import json
from typing import List

from monitoring.metrics import get_metrics_collector, MetricNames
from monitoring.health import get_health_checker
from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.monitoring_api")

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@router.get("/health")
async def get_health():
    """Get system health status."""
    try:
        checker = get_health_checker()
        health = await checker.run_all_checks()
        return JSONResponse(content=health.to_dict())
    except Exception as e:
        LOG.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
            }
        )


@router.get("/metrics")
async def get_metrics():
    """Get all current metrics."""
    try:
        collector = get_metrics_collector()
        summary = collector.get_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        LOG.error(f"Metrics fetch failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/metrics/{metric_name}")
async def get_metric_stats(metric_name: str):
    """Get statistics for a specific metric."""
    try:
        collector = get_metrics_collector()
        stats = collector.get_stats(metric_name)
        percentiles = collector.get_percentiles(metric_name)

        return JSONResponse(content={
            "name": metric_name,
            "stats": stats,
            "percentiles": percentiles,
        })
    except Exception as e:
        LOG.error(f"Metric stats fetch failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.get("/metrics/{metric_name}/timeseries")
async def get_metric_timeseries(metric_name: str, limit: int = 100):
    """Get time series data for a metric."""
    try:
        collector = get_metrics_collector()
        points = collector.get_time_series(metric_name)

        # Limit and format
        points = points[-limit:]
        series = [
            {
                "timestamp": p.timestamp,
                "value": p.value,
                "tags": p.tags,
            }
            for p in points
        ]

        return JSONResponse(content={
            "name": metric_name,
            "series": series,
        })
    except Exception as e:
        LOG.error(f"Time series fetch failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Send updates every 2 seconds
            await asyncio.sleep(2)

            collector = get_metrics_collector()
            summary = collector.get_summary()

            await websocket.send_json({
                "type": "metrics_update",
                "data": summary,
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        LOG.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
