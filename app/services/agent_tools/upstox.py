from __future__ import annotations

from typing import Optional, Dict, Any

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from app.stream_utils import emit_process
from app.utils.model import CandleQuery
from app.services.upstox import UpstoxProvider
from app.services.db.companies import (
	resolve_bse_scripcode,
	get_company_by_isin,
)


# ----------------- Helpers -----------------
def _resolve_instrument_key(
	*,
	instrument_key: Optional[str] = None,
	bse_code: Optional[str] = None,
	bse_symbol: Optional[str] = None,
	nse_symbol: Optional[str] = None,
	isin: Optional[str] = None,
	company_name: Optional[str] = None,
) -> Optional[str]:
	"""
	Try to resolve an Upstox instrument_key from various identifiers using the companies table.

	Resolution order:
	- If instrument_key provided, use as-is.
	- If NSE symbol provided or resolvable, assume format "NSE_EQ:{SYMBOL}".
	- Else resolve BSE scrip code via companies helpers and assume "BSE_EQ:{CODE}".

	Notes:
	- Upstox instrument formats vary. This uses a common heuristic (NSE_EQ/BSE_EQ prefix).
	  If your account uses a different scheme, pass instrument_key directly.
	"""
	if instrument_key:
		return instrument_key

	# Prefer NSE symbol when available
	if nse_symbol:
		return f"NSE_EQ:{nse_symbol}"

	# Try to infer NSE symbol from company row if only ISIN/company name provided
	if not nse_symbol and (isin or company_name):
		# Try via ISIN first
		if isin:
			try:
				row = get_company_by_isin(isin)
				if row and row.get("nse_symbol"):
					return f"NSE_EQ:{row['nse_symbol']}"
			except Exception:
				pass

	# Fallback to BSE scripcode path with fuzzy support
	bse = resolve_bse_scripcode(
		bse_code=bse_code,
		bse_symbol=bse_symbol,
		nse_symbol=nse_symbol,
		isin=isin,
		company_name=company_name,
	)
	if bse:
		return f"BSE_EQ:{bse}"

	return None


# ----------------- Tool implementation -----------------
def upstox_fetch_candles(
	*,
	unit: str,
	interval: str,
	from_date: str,
	to_date: str,
	instrument_key: Optional[str] = None,
	bse_code: Optional[str] = None,
	bse_symbol: Optional[str] = None,
	nse_symbol: Optional[str] = None,
	isin: Optional[str] = None,
	company_name: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	Fetch historical candles from Upstox, returning a canonical CandleSeries dict.

	Provide either instrument_key directly or any of the identifiers (nse_symbol, bse_code, bse_symbol, isin, company_name).
	Dates must be in the format expected by Upstox (e.g., YYYY-MM-DD or ISO datetime depending on unit/interval).
	"""
	emit_process({"message": "Resolving instrument"})
	key = _resolve_instrument_key(
		instrument_key=instrument_key,
		bse_code=bse_code,
		bse_symbol=bse_symbol,
		nse_symbol=nse_symbol,
		isin=isin,
		company_name=company_name,
	)
	if not key:
		return {"error": "Unable to resolve instrument_key from provided identifiers"}

	emit_process({"message": f"Fetching candles for {key}"})
	provider = UpstoxProvider()
	query = CandleQuery(
		instrument_key=key,
		unit=unit,  # type: ignore[arg-type]
		interval=interval,
		from_date=from_date,
		to_date=to_date,
	)
	try:
		raw = provider.fetch_historical_candles(query)
		series = UpstoxProvider.convert_upstox_candles_to_canonical(raw, query)
		# Return a plain dict for JSON-serializable tool output
		return {"candles": series.model_dump()}
	except Exception as e:
		return {"error": f"Upstox request failed: {e}"}


# ----------------- Structured tool wrapper -----------------
class UpstoxCandlesInput(BaseModel):
	unit: str = Field(..., description="Time unit: minutes/hours/days/weeks/months")
	interval: str = Field(..., description="Interval such as 1, 5, 15, 60, 1D, etc.")
	from_date: str = Field(..., description="Start date/time (Upstox format: YYYY-MM-DD or ISO)")
	to_date: str = Field(..., description="End date/time (Upstox format: YYYY-MM-DD or ISO)")
	instrument_key: Optional[str] = Field(None, description="Upstox instrument_key if known (e.g., NSE_EQ:RELIANCE)")
	bse_code: Optional[str] = Field(None, description="BSE scrip code (e.g., 500325)")
	bse_symbol: Optional[str] = Field(None, description="BSE symbol")
	nse_symbol: Optional[str] = Field(None, description="NSE trading symbol (e.g., RELIANCE)")
	isin: Optional[str] = Field(None, description="ISIN to resolve the company")
	company_name: Optional[str] = Field(None, description="Company name for fuzzy resolution")


UPSTOX_FETCH_CANDLES_TOOL = StructuredTool.from_function(
	func=upstox_fetch_candles,
	name="upstox_fetch_candles",
	description=(
		"Fetch historical candle data from Upstox. Provide instrument_key or identifiers like "
		"nse_symbol/bse_code/isin/company_name; fuzzy resolution is attempted via the companies table."
	),
	args_schema=UpstoxCandlesInput,
)
