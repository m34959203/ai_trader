from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from tasks import reports as reports_tasks

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/status")
async def reports_status() -> Dict[str, Any]:
    return reports_tasks.get_reports_summary()


@router.post("/generate")
async def reports_generate(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    return await reports_tasks.generate_reports(session=session)


@router.get("/download/{kind}")
async def reports_download(kind: str) -> FileResponse:
    if kind not in {"csv", "pdf"}:
        raise HTTPException(status_code=404, detail="unsupported_report_type")
    path = reports_tasks.get_last_report_path(kind)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="report_not_found")
    media_type = "text/csv" if kind == "csv" else "application/pdf"
    return FileResponse(path, filename=path.name, media_type=media_type)
