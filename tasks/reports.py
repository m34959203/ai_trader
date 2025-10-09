import asyncio
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import AsyncSessionLocal

try:  # pragma: no cover - optional DB dependency
    from db import crud_orders
except Exception:  # pragma: no cover - fallback when DB is not configured
    crud_orders = None  # type: ignore

LOG = logging.getLogger("ai_trader.reports")

REPORTS_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_REPORT_LOCK = asyncio.Lock()
_LAST_REPORTS: Dict[str, Any] = {}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _fetch_orders(limit: int, session: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
    if crud_orders is None:
        return []
    if session is not None:
        return await crud_orders.get_last_orders(session, limit=limit)
    async with AsyncSessionLocal() as local_session:
        return await crud_orders.get_last_orders(local_session, limit=limit)


def _compose_summary(df: pd.DataFrame, generated_at: datetime) -> str:
    lines = [f"Live trade report generated {generated_at.isoformat(timespec='seconds')}", ""]
    if df.empty:
        lines.append("No trades captured in the selected window.")
        return "\n".join(lines)
    columns = [c for c in ["created_at", "symbol", "side", "status", "qty", "filled_qty", "price"] if c in df.columns]
    head = df[columns].head(15)
    for _, row in head.iterrows():
        parts = []
        for col in columns:
            value = row.get(col)
            parts.append(f"{col}={value}")
        lines.append(" ".join(parts))
    if len(df) > len(head):
        lines.append(f"... total rows: {len(df)}")
    return "\n".join(lines)


def _safe_encode(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _simple_pdf(text: str, path: Path) -> None:
    lines = text.splitlines() or [""]
    content_lines = ["BT", "/F1 10 Tf", "50 780 Td"]
    for idx, line in enumerate(lines):
        safe = _safe_encode(line)
        safe = safe.encode("latin-1", "replace").decode("latin-1")
        if idx == 0:
            content_lines.append(f"({safe}) Tj")
        else:
            content_lines.append("T*")
            content_lines.append(f"({safe}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1")

    buffer = io.BytesIO()

    def write(payload: Any) -> None:
        if isinstance(payload, bytes):
            buffer.write(payload)
        else:
            buffer.write(str(payload).encode("latin-1"))

    write("%PDF-1.4\n")
    offsets: List[int] = []

    def add_object(body: str) -> None:
        offsets.append(buffer.tell())
        write(body)

    add_object("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    add_object("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    add_object("3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n")

    offsets.append(buffer.tell())
    write(f"4 0 obj << /Length {len(stream)} >> stream\n")
    write(stream)
    write("\nendstream\nendobj\n")

    offsets.append(buffer.tell())
    write("5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")

    xref_pos = buffer.tell()
    write("xref\n0 6\n0000000000 65535 f \n")
    for off in offsets:
        write(f"{off:010d} 00000 n \n")
    write("trailer << /Size 6 /Root 1 0 R >>\n")
    write(f"startxref\n{xref_pos}\n%%EOF\n")

    path.write_bytes(buffer.getvalue())


async def generate_reports(limit: int = 200, session: Optional[AsyncSession] = None) -> Dict[str, Any]:
    async with _REPORT_LOCK:
        orders = await _fetch_orders(limit, session=session)
        df = pd.DataFrame(orders)
        timestamp = _utc_now()
        if not df.empty:
            df = df.sort_values("created_at", ascending=False)
        csv_path = REPORTS_DIR / f"live_orders_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.csv"
        df.to_csv(csv_path, index=False)

        summary_text = _compose_summary(df, timestamp)
        pdf_path = REPORTS_DIR / (csv_path.stem + ".pdf")
        _simple_pdf(summary_text, pdf_path)

        csv_size = csv_path.stat().st_size if csv_path.exists() else 0
        pdf_size = pdf_path.stat().st_size if pdf_path.exists() else 0

        global _LAST_REPORTS
        _LAST_REPORTS = {
            "generated_at": timestamp.isoformat(),
            "csv": {
                "path": str(csv_path),
                "size": csv_size,
                "generated_at": timestamp.isoformat(),
            },
            "pdf": {
                "path": str(pdf_path),
                "size": pdf_size,
                "generated_at": timestamp.isoformat(),
            },
        }
        LOG.info("Generated live trade reports: csv=%s bytes=%s pdf=%s bytes=%s", csv_path.name, csv_size, pdf_path.name, pdf_size)
        return dict(_LAST_REPORTS)


def _latest_file(suffix: str) -> Optional[Path]:
    candidates = sorted(REPORTS_DIR.glob(f"*.{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def get_last_report_path(kind: str) -> Optional[Path]:
    entry = _LAST_REPORTS.get(kind)
    if isinstance(entry, dict) and entry.get("path"):
        path = Path(entry["path"])
        if path.exists():
            return path
    suffix = "csv" if kind == "csv" else "pdf"
    return _latest_file(suffix)


def get_reports_summary() -> Dict[str, Any]:
    summary = dict(_LAST_REPORTS)
    if "csv" in summary:
        summary["csv"] = dict(summary["csv"])
    if "pdf" in summary:
        summary["pdf"] = dict(summary["pdf"])
    return summary


async def background_loop(
    *,
    interval_minutes: int = 60,
    stop_event: Optional[asyncio.Event] = None,
    limit: int = 200,
) -> None:
    delay = max(1, int(interval_minutes)) * 60
    while True:
        if stop_event and stop_event.is_set():
            LOG.info("Reports background loop stopped via stop_event")
            break
        try:
            await generate_reports(limit=limit)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOG.warning("Scheduled report generation failed: %r", exc)
        await asyncio.sleep(delay)
