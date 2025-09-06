from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.utils.stream_utils import emit_process


class ChartSpec(BaseModel):
    type: str = Field(..., description="Chart type")
    title: str = Field(..., description="Chart title")
    description: str = Field(..., description="Chart description")
    dataKey: str = Field(..., description="Field name for chart values")
    nameKey: str = Field(..., description="Field name for chart labels")
    data: List[Dict[str, Any]] = Field(..., description="List of data objects with nameKey and dataKey fields")


# ------------------ Dynamic multi-type emitter ------------------

# Minimal schema registry: required keys per chart type
CHART_SCHEMAS: Dict[str, List[str]] = {
    # Bars
    "bar-standard": ["type", "title", "description", "dataKey", "nameKey", "data"],
    "bar-multiple": ["type", "title", "description", "dataKey", "nameKey", "xAxisKey", "layout", "data"],
    "bar-stacked": ["type", "title", "description", "xAxisKey", "nameKey", "groupedKeys", "data"],
    "bar-negative": ["type", "title", "description", "xAxisKey", "groupedKeys", "legend", "data"],

    # Areas
    "area-standard": ["type", "title", "description", "dataKey", "nameKey", "color", "data"],
    "area-linear": ["type", "title", "description", "dataKey", "nameKey", "color", "data"],
    "area-stacked": ["type", "title", "description", "dataKey", "groupedKeys", "color", "legend", "data"],
    # alternative area-stacked variant using xAxisKey instead of dataKey
    "area-stacked-alt": ["type", "title", "description", "groupedKeys", "xAxisKey", "data"],

    # Lines
    "line-standard": ["type", "title", "description", "dataKey", "nameKey", "color", "data"],
    "line-linear": ["type", "title", "description", "dataKey", "nameKey", "color", "legend", "data"],
    "line-multiple": ["type", "title", "description", "dataKey", "groupedKeys", "color", "legend", "data"],
    "line-dots": ["type", "title", "description", "dataKey", "color", "legend", "dots", "data"],
    "line-label": ["type", "title", "description", "dataKey", "xAxisKey", "customLabel", "data"],
    "line-label-2": ["type", "title", "description", "dataKey", "nameKey", "color", "label", "data"],

    # Pies / Radials / Radar
    "pie-standard": ["type", "title", "description", "dataKey", "nameKey", "data"],
    "pie-label": ["type", "title", "description", "dataKey", "nameKey", "label", "legend", "data"],
    "pie-interactive": ["type", "title", "description", "dataKey", "nameKey", "legend", "data"],
    "pie-donut": ["type", "title", "description", "dataKey", "nameKey", "legend", "data"],

    "radar-standard": ["type", "title", "description", "dataKey", "nameKey", "color", "legend", "data"],
    "radar-lines-only": ["type", "title", "description", "dataKey", "nameKey", "color", "legend", "data"],
    "radar-multiple": ["type", "title", "description", "dataKey", "color", "legend", "grid", "data"],

    "radial-standard": ["type", "title", "description", "dataKey", "groupedKeys", "color", "legend", "data"],
    "radial-stacked": ["type", "title", "description", "groupedKeys", "nameKey", "data"],
    "radial-progress": ["type", "title", "description", "dataKey", "nameKey", "centerText", "centerLabel", "data"],
}


class ChartPayloadInput(BaseModel):
    payload: Dict[str, Any]


def _validate_payload(payload: Dict[str, Any]) -> None:
    chart_type = str(payload.get("type", "")).strip()
    if not chart_type:
        raise ValueError("Missing 'type' in chart payload")
    required = CHART_SCHEMAS.get(chart_type)
    if not required:
        raise ValueError(f"Unsupported chart type: {chart_type}")
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



