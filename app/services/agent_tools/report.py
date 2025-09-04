from __future__ import annotations

from typing import Optional, Dict, Any, List

from app.stream_utils import emit_process
from app.services.report import IngestorService


def _summarize_result(kind: str, payload: Dict[str, Any]) -> str:
    """Return a concise, human-readable summary for the agent output."""
    try:
        scrip = payload.get("scripcode") or "?"
        time_frame = payload.get("timeFrame") or f"{payload.get('fromYear')}-{payload.get('toYear')}"
        filtered = payload.get("filtered_table_count")
        total = payload.get("raw_table_count")
        stored_n = int(payload.get("stored_files_count") or 0)
        stored_dir = payload.get("stored_files_dir") or ""
        status = payload.get("status") or ""
        links_n = len(payload.get("xbrl_links") or [])
        parts = [
            f"{kind}: scripcode {scrip}",
            f"range {time_frame}",
            f"rows {filtered}/{total}" if filtered is not None and total is not None else None,
            f"links {links_n}",
            f"saved {stored_n} file(s)" + (f" to {stored_dir}" if stored_n and stored_dir else ""),
            f"status: {status}",
        ]
        return ", ".join([p for p in parts if p])
    except Exception:
        return f"{kind}: done"


# ---------- Tool call functions ----------
def bse_shp_extract(
    scripcode: Optional[str] = None,
    from_year: int = 0,
    to_year: int = 0,
    stock_query: Optional[str] = None,
    max_xbrl_files: int = 3,
) -> str:
    """
    Extract BSE Shareholding Pattern XBRL files for a scripcode or stock/company query within a year range.
    Returns a short summary string with counts and save location.
    """
    emit_process({"message": "Extracting Shareholding Pattern from BSE"})
    svc = IngestorService()
    res = svc.extract_shareholding_pattern(
        scripcode=scripcode,
        from_year=from_year,
        to_year=to_year,
        stock_query=stock_query,
        max_xbrl_files=max_xbrl_files,
    )
    return _summarize_result("BSE SHP", res)


def bse_cg_extract_range(
    scripcode: Optional[str] = None,
    from_year: int = 0,
    to_year: int = 0,
    stock_query: Optional[str] = None,
    max_xbrl_files: int = 10,
) -> str:
    """
    Discover and download Corporate Governance XBRL files for the given year range.
    Returns a short summary string with counts and save location.
    """
    emit_process({"message": "Extracting Corporate Governance reports from BSE"})
    svc = IngestorService()
    res = svc.extract_corporate_governance_range(
        scripcode=scripcode,
        from_year=from_year,
        to_year=to_year,
        stock_query=stock_query,
        max_xbrl_files=max_xbrl_files,
    )
    return _summarize_result("BSE CG", res)


def bse_ar_extract(
    scripcode: Optional[str] = None,
    from_year: int = 0,
    to_year: int = 0,
    stock_query: Optional[str] = None,
    max_files: int = 5,
) -> str:
    """
    Download Annual Report PDFs from BSE for the given year range.
    Returns a short summary string with counts and save location.
    """
    emit_process({"message": "Extracting Annual Reports (PDFs) from BSE"})
    svc = IngestorService()
    res = svc.extract_annual_results(
        scripcode=scripcode,
        from_year=from_year,
        to_year=to_year,
        stock_query=stock_query,
        max_files=max_files,
    )
    try:
        scrip = res.get("scripcode") or "?"
        time_frame = res.get("timeFrame") or f"{res.get('fromYear')}-{res.get('toYear')}"
        stored_n = int(res.get("stored_files_count") or 0)
        stored_dir = res.get("stored_files_dir") or ""
        status = res.get("status") or ""
        links_n = len(res.get("pdf_links") or [])
        parts = [
            f"BSE AR: scripcode {scrip}",
            f"range {time_frame}",
            f"pdfs {links_n}",
            f"saved {stored_n} file(s)" + (f" to {stored_dir}" if stored_n and stored_dir else ""),
            f"status: {status}",
        ]
        return ", ".join([p for p in parts if p])
    except Exception:
        return "BSE AR: done"


# ---------- Structured tools (LangChain) ----------
try:
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool

    class SHPInput(BaseModel):
        scripcode: Optional[str] = Field(None, description="BSE scripcode, e.g., 526612")
        from_year: int = Field(..., ge=2000, le=2100, description="Start year (inclusive)")
        to_year: int = Field(..., ge=2000, le=2100, description="End year (inclusive)")
        stock_query: Optional[str] = Field(None, description="Company name to resolve scripcode if unknown")
        max_xbrl_files: int = Field(3, ge=1, le=20, description="Max files to download")

    class CGInput(BaseModel):
        scripcode: Optional[str] = Field(None, description="BSE scripcode, e.g., 526612")
        from_year: int = Field(..., ge=2000, le=2100, description="Start year (inclusive)")
        to_year: int = Field(..., ge=2000, le=2100, description="End year (inclusive)")
        stock_query: Optional[str] = Field(None, description="Company name to resolve scripcode if unknown")
        max_xbrl_files: int = Field(10, ge=1, le=40, description="Max files to download")

    class ARInput(BaseModel):
        scripcode: Optional[str] = Field(None, description="BSE scripcode, e.g., 526612")
        from_year: int = Field(..., ge=2000, le=2100, description="Start year (inclusive)")
        to_year: int = Field(..., ge=2000, le=2100, description="End year (inclusive)")
        stock_query: Optional[str] = Field(None, description="Company name to resolve scripcode if unknown")
        max_files: int = Field(5, ge=1, le=40, description="Max files to download")

    BSE_SHP_EXTRACT_TOOL = StructuredTool.from_function(
        func=bse_shp_extract,
        name="bse_shp_extract",
        description=(
            "Extract Shareholding Pattern (SHP) XBRL files from BSE for a given company/year range. "
            "Use when the user asks about shareholding pattern or SHP."
        ),
        args_schema=SHPInput,
    )

    BSE_CG_EXTRACT_TOOL = StructuredTool.from_function(
        func=bse_cg_extract_range,
        name="bse_cg_extract",
        description=(
            "Extract Corporate Governance (CG) XBRL files from BSE for a given company/year range. "
            "Use when the user asks about corporate governance reports."
        ),
        args_schema=CGInput,
    )

    BSE_AR_EXTRACT_TOOL = StructuredTool.from_function(
        func=bse_ar_extract,
        name="bse_ar_extract",
        description=(
            "Extract Annual Report (AR) PDFs from BSE for a given company/year range. "
            "Use when the user asks about annual reports or AR PDFs."
        ),
        args_schema=ARInput,
    )
except Exception:
    BSE_SHP_EXTRACT_TOOL = None
    BSE_CG_EXTRACT_TOOL = None
    BSE_AR_EXTRACT_TOOL = None
