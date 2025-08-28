"""Utilities for chart schema and mapping.

Loads the JSON definitions from chart.json and exposes constants and helpers.
Also defines a default candlestick format used by Yahoo candles.
"""
from __future__ import annotations
import json
import os
from typing import Any, Dict

_DIR = os.path.dirname(__file__)
_JSON_PATH = os.path.join(_DIR, "chart.json")

CHART_MAPPINGS: Dict[str, str] = {}
SUPPORTED_CHART_FORMATS: Dict[str, Dict[str, Any]] = {}

try:
    with open(_JSON_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    # chart.json currently contains two top-level JSON objects appended; parse both
    parts = [p.strip() for p in raw.split("}\n\n{")]
    if len(parts) == 2:
        first = json.loads(parts[0] + "}")
        second = json.loads("{" + parts[1])
        CHART_MAPPINGS = first.get("chart_mappings", {})
        SUPPORTED_CHART_FORMATS = second.get("SUPPORTED_CHART_FORMATS", {})
    else:
        doc = json.loads(raw)
        CHART_MAPPINGS = doc.get("chart_mappings", {})
        SUPPORTED_CHART_FORMATS = doc.get("SUPPORTED_CHART_FORMATS", {})
except Exception:
    CHART_MAPPINGS = {}
    SUPPORTED_CHART_FORMATS = {}

# Provide a local definition for candlestick since chart.json may not include it yet
if "candlestick-standard" not in SUPPORTED_CHART_FORMATS:
    SUPPORTED_CHART_FORMATS["candlestick-standard"] = {
        "required_keys": [
            "type",
            "title",
            "description",
            "xAxisKey",
            "groupedKeys",
            "nameKey",
            "data",
        ],
        "example": {
            "type": "candlestick-standard",
            "title": "Price Candles",
            "description": "OHLC candles",
            "xAxisKey": "time",
            "groupedKeys": ["open", "high", "low", "close"],
            "nameKey": "time",
            "data": [],
        },
    }
