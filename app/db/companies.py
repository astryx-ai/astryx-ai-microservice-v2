from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, TypedDict
from app.db.supabase import get_supabase_client


class CompanyRow(TypedDict, total=False):
	id: str
	company_name: str
	nse_symbol: Optional[str]
	bse_code: Optional[str]
	bse_symbol: Optional[str]
	isin: Optional[str]
	industry: Optional[str]
	status: Optional[str]
	market_cap: Optional[float]
	created_at: str
	updated_at: str


def _companies_table():
	"""Return a Supabase query object for the companies table."""
	return get_supabase_client().table("companies")


def get_company_by_isin(isin: str) -> Optional[CompanyRow]:
	"""Fetch a single company by ISIN."""
	if not isin:
		return None
	res = _companies_table().select("*").eq("isin", isin).limit(1).execute()
	data = getattr(res, "data", None) or []
	return (data[0] if data else None)  # type: ignore[return-value]


def get_company_by_bse_code(bse_code: str) -> Optional[CompanyRow]:
	"""Fetch a single company by BSE scrip code."""
	if not bse_code:
		return None
	res = _companies_table().select("*").eq("bse_code", bse_code).limit(1).execute()
	data = getattr(res, "data", None) or []
	return (data[0] if data else None)  # type: ignore[return-value]


def get_company_by_bse_symbol(bse_symbol: str) -> Optional[CompanyRow]:
	"""Fetch a single company by BSE symbol."""
	if not bse_symbol:
		return None
	res = _companies_table().select("*").eq("bse_symbol", bse_symbol).limit(1).execute()
	data = getattr(res, "data", None) or []
	return (data[0] if data else None)  # type: ignore[return-value]


def get_company_by_nse_symbol(nse_symbol: str) -> Optional[CompanyRow]:
	"""Fetch a single company by NSE symbol."""
	if not nse_symbol:
		return None
	res = _companies_table().select("*").eq("nse_symbol", nse_symbol).limit(1).execute()
	data = getattr(res, "data", None) or []
	return (data[0] if data else None)  # type: ignore[return-value]


def search_companies(query: str, limit: int = 20) -> List[CompanyRow]:
	"""Search by company_name ilike query% and also by symbols equality best-effort."""
	if not query:
		return []
	q = query.strip()
	# Try name search first
	builder = _companies_table().select("*").ilike("company_name", f"%{q}%").limit(limit)
	res = builder.execute()
	data = list(getattr(res, "data", None) or [])
	# If empty, try direct symbol matches
	if not data:
		res2 = (
			_companies_table()
			.select("*")
			.or_(
				f"bse_code.eq.{q},bse_symbol.eq.{q},nse_symbol.eq.{q},isin.eq.{q}"
			)
			.limit(limit)
			.execute()
		)
		data = list(getattr(res2, "data", None) or [])
	return data  # type: ignore[return-value]


def fuzzy_search_companies(query: str, limit: int = 10) -> List[CompanyRow]:
	"""Fuzzy search using the companies_fuzzy_search RPC (pg_trgm based).

	Requires the SQL in app/services/db/fuzzy.sql to be applied to the DB.
	"""
	if not query:
		return []
	supa = get_supabase_client()
	res = supa.rpc("companies_fuzzy_search", {"q": query, "limit_count": limit}).execute()
	return list(getattr(res, "data", None) or [])  # type: ignore[return-value]


def list_companies(
	*,
	status: Optional[str] = None,
	industry: Optional[str] = None,
	has_bse: Optional[bool] = None,
	has_nse: Optional[bool] = None,
	limit: int = 100,
) -> List[CompanyRow]:
	"""List companies with optional filters."""
	builder = _companies_table().select("*")
	if status:
		builder = builder.eq("status", status)
	if industry:
		builder = builder.eq("industry", industry)
	if has_bse is True:
		builder = builder.not_.is_("bse_code", "null")
	elif has_bse is False:
		builder = builder.is_("bse_code", "null")
	if has_nse is True:
		builder = builder.not_.is_("nse_symbol", "null")
	elif has_nse is False:
		builder = builder.is_("nse_symbol", "null")
	res = builder.limit(limit).execute()
	return list(getattr(res, "data", None) or [])  # type: ignore[return-value]


def resolve_bse_scripcode(
	*,
	bse_code: Optional[str] = None,
	bse_symbol: Optional[str] = None,
	nse_symbol: Optional[str] = None,
	isin: Optional[str] = None,
	company_name: Optional[str] = None,
) -> Optional[str]:
	"""Resolve BSE scripcode from any provided identifier via companies table."""
	# Strict priority: direct bse_code, else bse_symbol, then ISIN, NSE, then name search
	if bse_code:
		return bse_code
	if bse_symbol:
		row = get_company_by_bse_symbol(bse_symbol)
		return row.get("bse_code") if row else None
	if isin:
		row = get_company_by_isin(isin)
		return row.get("bse_code") if row else None
	if nse_symbol:
		row = get_company_by_nse_symbol(nse_symbol)
		return row.get("bse_code") if row else None
	if company_name:
		matches = search_companies(company_name, limit=1)
		if matches:
			return matches[0].get("bse_code")
		# Fuzzy fallback if exact/ILIKE search didn't return anything
		fuzzy = fuzzy_search_companies(company_name, limit=1)
		if fuzzy:
			return fuzzy[0].get("bse_code")
	return None