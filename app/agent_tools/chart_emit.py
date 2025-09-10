from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.utils.stream_utils import emit_process
from .chart_formats import SUPPORTED_CHART_FORMATS
    

class ChartSpec(BaseModel):
    type: str = Field(..., description="Chart type")
    title: str = Field(..., description="Chart title")
    description: str = Field(..., description="Chart description")
    dataKey: str = Field(..., description="Field name for chart values")
    nameKey: str = Field(..., description="Field name for chart labels")
    data: List[Dict[str, Any]] = Field(..., description="List of data objects with nameKey and dataKey fields")


# ------------------ Dynamic multi-type emitter ------------------


class ChartPayloadInput(BaseModel):
    payload: Dict[str, Any]


def _validate_payload(payload: Dict[str, Any]) -> None:
    chart_type = str(payload.get("type", "")).strip()
    if not chart_type:
        raise ValueError("Missing 'type' in chart payload")
    schema = SUPPORTED_CHART_FORMATS.get(chart_type)
    if not schema:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    required = schema.get("required_keys", [])
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required keys for {chart_type}: {missing}")
    # Basic data sanity
    if "data" in required and not isinstance(payload.get("data"), list):
        raise ValueError("'data' must be a list")


def emit_chart(payload: Dict[str, Any]) -> str:
    """Validate arbitrary chart payload against known schemas and emit as chart_data event.
    This tool can be called multiple times within a single agent run to emit multiple charts.
    Returns empty string so it doesn't pollute agent text output.
    """
    try:
        _validate_payload(payload)
        emit_process({"event": "chart_data", "chart": payload})
        print(f"[ChartEmit] Emitted dynamic chart: {payload.get('title', 'Untitled')} ({payload.get('type')})")
    except Exception as e:
        print(f"[ChartEmit] Failed to emit dynamic chart: {e}")
    return ""



