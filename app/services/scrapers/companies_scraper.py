import pandas as pd
import requests
from io import StringIO
from typing import Dict, List, Optional


_UA = {
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
}


def _safe_str(x: Optional[str]) -> Optional[str]:
	if x is None:
		return None
	t = str(x).strip()
	return t if t else None


def _to_number(x: Optional[str]) -> Optional[float]:
	if x is None:
		return None
	t = str(x).strip().replace(",", "")
	if t == "" or t == "-":
		return None
	try:
		return float(t)
	except Exception:
		return None


def fetch_nse_companies() -> List[Dict]:
	url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
	r = requests.get(url, headers=_UA)
	r.raise_for_status()
	df = pd.read_csv(StringIO(r.text))

	# Normalize header names to simplify mapping
	cols = {c.lower().strip(): c for c in df.columns}

	def col(*names: str) -> Optional[str]:
		for n in names:
			if n in cols:
				return cols[n]
		return None

	# Keep only Active equities: SERIES == 'EQ', optionally Status == Active
	series_col = col("series")
	if series_col and series_col in df.columns:
		df = df[df[series_col].astype(str).str.upper() == "EQ"]
	status_col = col("status")
	if status_col and status_col in df.columns:
		df = df[df[status_col].astype(str).str.lower() == "active"]

	out: List[Dict] = []
	for _, row in df.iterrows():
		symbol = row.get(col("symbol"))
		name = row.get(col("name of company", "name_of_company", "company name", "company_name"))
		isin = row.get(col("isin number", "isin", "isin_no", "isin code"))
		status = row.get(col("status"))
		industry = row.get(col("industry"))
		rec = {
			"company_name": _safe_str(name),
			"nse_symbol": _safe_str(symbol),
			"bse_code": None,
			"bse_symbol": None,
			"isin": _safe_str(isin),
			"industry": _safe_str(industry),
			"status": _safe_str(status),
			"market_cap": None,
		}
		out.append(rec)
	return out


def fetch_bse_companies() -> List[Dict]:
	url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status="
	headers = {
		"accept": "application/json, text/plain, */*",
		"accept-encoding": "gzip, deflate, br, zstd",
		"accept-language": "en-US,en;q=0.9",
		"origin": "https://www.bseindia.com",
		"referer": "https://www.bseindia.com/",
		"sec-ch-ua": '"Not;A=Brand";v="99", "Microsoft Edge";v="139", "Chromium";v="139"',
		"sec-ch-ua-mobile": "?0",
		"sec-ch-ua-platform": '"Windows"',
		"sec-fetch-dest": "empty",
		"sec-fetch-mode": "cors",
		"sec-fetch-site": "same-site",
		"user-agent": _UA["User-Agent"],
	}
	r = requests.get(url, headers=headers)
	r.raise_for_status()
	data = r.json()
	if not isinstance(data, list):
		return []

	out: List[Dict] = []
	for it in data:
		# Keep only Active equities. Some payloads omit 'Segment' even when the endpoint is filtered by segment=Equity.
		status_val = str(it.get("Status") or it.get("status") or it.get("Active") or "").strip().lower()
		seg_raw = it.get("Segment") if ("Segment" in it or "segment" in it) else None
		segment_val = str(it.get("Segment") or it.get("segment") or "").strip().lower()
		status_ok = (status_val == "active")
		segment_ok = (segment_val == "equity") if seg_raw is not None else True
		if not (status_ok and segment_ok):
			continue
		# Try multiple key variants commonly seen in BSE payloads
		code = it.get("SCRIP_CD") or it.get("scrip_cd") or it.get("Scripcode") or it.get("ScripCode") or it.get("SC_CODE")
		# Security Id is the correct BSE symbol; strip trailing '#'
		sec_id = it.get("SecurityId") or it.get("securityId") or it.get("ScripID") or it.get("Scrip_Id") or it.get("scrip_id")
		if isinstance(sec_id, str):
			sec_id = sec_id.replace("#", "").strip()
		# Security Name for fallback display only
		sec_name = it.get("Scrip_Name") or it.get("SC_NAME") or it.get("sc_name")
		name = it.get("SC_FULLNAME") or it.get("sc_fullname") or it.get("Issuer_Name") or it.get("FullName") or it.get("fullName")
		isin = it.get("ISINNO") or it.get("ISINNo") or it.get("isinno") or it.get("ISIN") or it.get("isin")
		industry = it.get("INDUSTRY") or it.get("industry")
		status = it.get("Status") or it.get("Active") or it.get("active") or it.get("status")
		mcap = it.get("MktcapFull") or it.get("mktcapFull") or it.get("Mktcap") or it.get("mktcap")

		rec = {
			"company_name": _safe_str(name) or _safe_str(sec_name),
			"nse_symbol": None,
			"bse_code": _safe_str(code),
			"bse_symbol": _safe_str(sec_id),
			"isin": _safe_str(isin),
			"industry": _safe_str(industry),
			"status": _safe_str(str(status) if status is not None else None),
			"market_cap": _to_number(mcap),
		}
		out.append(rec)
	return out


def get_companies() -> List[Dict]:
	"""
	Merge NSE and BSE lists by ISIN where available. Prefer BSE industry/market_cap,
	NSE company_name/nse_symbol when merging.
	Only return records that have an ISIN to satisfy upsert uniqueness.
	"""
	nse = fetch_nse_companies()
	bse = fetch_bse_companies()

	def _norm_name(s: Optional[str]) -> Optional[str]:
		if not s:
			return None
		import re
		t = s.lower()
		t = re.sub(r"[^a-z0-9\s]", " ", t)
		# remove common suffixes that differ between exchanges
		t = re.sub(r"\b(ltd|limited|india|industries|co|corp|corporation|the)\b", " ", t)
		t = re.sub(r"\s+", " ", t).strip()
		return t or None

	# Seed by ISIN using NSE (authoritative for EQ + ISIN)
	by_isin: Dict[str, Dict] = {}
	name_to_isin: Dict[str, str] = {}
	for rec in nse:
		isin = rec.get("isin")
		if not isin:
			continue
		by_isin[isin] = rec.copy()
		key = _norm_name(rec.get("company_name"))
		if key:
			# If duplicate names map to different ISINs, keep the first (most common case is unique)
			name_to_isin.setdefault(key, isin)

	# 1) Merge by ISIN if BSE ever provides it
	for rec in bse:
		isin = rec.get("isin")
		if not isin:
			continue
		base = by_isin.get(isin, {
			"company_name": None,
			"nse_symbol": None,
			"bse_code": None,
			"bse_symbol": None,
			"isin": isin,
			"industry": None,
			"status": None,
			"market_cap": None,
		})
		base["company_name"] = base.get("company_name") or rec.get("company_name")
		base["bse_code"] = base.get("bse_code") or rec.get("bse_code")
		base["bse_symbol"] = base.get("bse_symbol") or rec.get("bse_symbol")
		base["industry"] = rec.get("industry") or base.get("industry")
		base["status"] = rec.get("status") or base.get("status")
		base["market_cap"] = rec.get("market_cap") if rec.get("market_cap") is not None else base.get("market_cap")
		by_isin[isin] = base

	# 2) Name-based augmentation for BSE records without ISIN
	for rec in bse:
		if rec.get("isin"):
			continue
		cand_keys = [
			_norm_name(rec.get("company_name")),
			_norm_name(rec.get("bse_symbol")),
		]
		cand_keys = [k for k in cand_keys if k]
		matched_isin: Optional[str] = None
		for k in cand_keys:
			if k in name_to_isin:
				matched_isin = name_to_isin[k]
				break
		if not matched_isin:
			continue
		base = by_isin.get(matched_isin)
		if not base:
			continue
		# augment NSE record with BSE identifiers
		if not base.get("bse_code") and rec.get("bse_code"):
			base["bse_code"] = rec.get("bse_code")
		if not base.get("bse_symbol") and rec.get("bse_symbol"):
			base["bse_symbol"] = rec.get("bse_symbol")
		# Prefer BSE fundamentals if available
		base["industry"] = rec.get("industry") or base.get("industry")
		base["status"] = rec.get("status") or base.get("status")
		base["market_cap"] = rec.get("market_cap") if rec.get("market_cap") is not None else base.get("market_cap")

	# Return all NSE-backed ISIN rows
	return [v for v in by_isin.values() if v.get("isin")]
