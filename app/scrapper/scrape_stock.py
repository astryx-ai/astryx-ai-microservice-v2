from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import random
from typing import Dict, Optional
import httpx


UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.moneycontrol.com/",
    "Upgrade-Insecure-Requests": "1",
}

def _client(extra_headers: Optional[dict] = None) -> httpx.Client:
    headers = BASE_HEADERS.copy()
    headers["User-Agent"] = random.choice(UA_POOL)
    if extra_headers:
        headers.update(extra_headers)
    return httpx.Client(http2=True, headers=headers, timeout=20, follow_redirects=True)

def scrape_company_stock(company_name):
    """Compatibility wrapper for external callers (fetch_stock.py)."""
    return scrape_company(company_name)  # noqa: F821

def normalize_number(val):
    if not val:
        return None
    text = str(val).replace(",", "").strip()
    # Convert 'Cr' or 'Lac' into pure numbers (keeping scale in crores)
    multiplier = 1
    if "Cr" in text:
        multiplier = 1
        text = re.sub(r"[^\d\.\-]", "", text)
    elif "Lac" in text:
        multiplier = 0.01
        text = re.sub(r"[^\d\.\-]", "", text)
    else:
        text = re.sub(r"[^\d\.\-]", "", text)
    try:
        return float(text) * multiplier
    except:
        return None

def find_company_url(company_name: str) -> Optional[str]:
    """Use Moneycontrol autosuggestion API to resolve the company overview URL.
    Prefer strong token-overlap matches to avoid wrong companies.
    """
    if not company_name:
        return None
    api = "https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php"
    params = {"classic": "true", "query": company_name, "type": 1, "format": "json"}
    try:
        with _client() as c:
            r = c.get(api, params=params)
            if r.status_code != 200:
                return None
            data = r.json()
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None

    # Token overlap ranking
    def norm_tokens(s: str):
        import re
        t = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
        return set(p for p in t.split() if p)

    target = norm_tokens(company_name)
    best_link = None
    best_score = 0.0
    for item in data:
        link = item.get("link_src") or item.get("link") or ""
        label = item.get("label") or item.get("name") or item.get("value") or ""
        if not link:
            continue
        cand = norm_tokens(label)
        if not cand:
            continue
        inter = len(target & cand)
        union = len(target | cand)
        score = (inter / union) if union else 0.0
        if "/india/stockpricequote/" in link:
            score += 0.2  # prefer stock pages
        if score > best_score:
            best_score = score
            best_link = link
    if best_link:
        return best_link
    return None

def _clean_label(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().replace(":", ""))

def _detect_unit(raw: str) -> str:
    t = (raw or "").lower()
    if "%" in t:
        return "%"
    if "cr" in t or "crore" in t:
        return "Cr"
    if "lac" in t or "lakh" in t:
        return "Lac"
    if "₹" in t or " rs" in t or "rs." in t:
        return "₹"
    return ""

def _to_number(raw: str) -> Optional[float]:
    """Parse textual numeric values. Convert Lac to Cr scale (Lac * 0.01)."""
    if raw is None:
        return None
    t = raw.strip()
    if t == "" or t == "-":
        return None
    mult = 1.0
    tl = t.lower()
    if "lac" in tl or "lakh" in tl:
        mult = 0.01  # 1 Lac = 0.01 Cr
    num = re.sub(r"[^\d\.\-]", "", t)
    try:
        return float(num) * mult
    except Exception:
        return None

def _map_metric(label: str) -> Optional[str]:
    key = _clean_label(label)
    mapping = {
        "open": "open_price",
        "previous close": "previous_close",
        "prev close": "previous_close",
        "high": "day_high",
        "low": "day_low",
        "volume": "volume",
        "value (lacs)": "turnover_lacs",
        "vwap": "vwap",
        "market cap": "market_cap",
        "mkt cap": "market_cap",
        "mkt cap (rs. cr.)": "market_cap",
        "beta": "beta",
        "uc limit": "upper_circuit",
        "lc limit": "lower_circuit",
        "52 week high": "week_52_high",
        "52 week low": "week_52_low",
        "face value": "face_value",
        "all time high": "all_time_high",
        "all time low": "all_time_low",
        "20d avg volume": "avg_volume_20d",
        "20d avg delivery(%)": "avg_delivery_20d",
        "book value per share": "book_value_per_share",
        "dividend yield": "dividend_yield",
        "ttm eps": "ttm_eps",
        "ttm pe": "ttm_pe",
        "p/b": "pb_ratio",
        "p/bv": "pb_ratio",
        "pb": "pb_ratio",
        "sector pe": "sector_pe",
        "pe": "pe_ratio",
        "p/e": "pe_ratio",
        "pe ratio": "pe_ratio",
    }
    for k, v in mapping.items():
        if k in key:
            return v
    return None

def _metric_desc(key: str) -> str:
    descriptions = {
        "current_price": "Current stock price",
        "open_price": "Opening price for the trading day",
        "previous_close": "Previous trading day closing price",
        "day_high": "Highest price during current trading day",
        "day_low": "Lowest price during current trading day",
        "volume": "Number of shares traded",
        "turnover_lacs": "Total value of shares traded in lacs",
        "vwap": "Volume weighted average price",
        "market_cap": "Total market capitalization in crores",
        "beta": "Stock volatility relative to market",
        "upper_circuit": "Maximum allowed price for the day",
        "lower_circuit": "Minimum allowed price for the day",
        "week_52_high": "Highest price in last 52 weeks",
        "week_52_low": "Lowest price in last 52 weeks",
        "face_value": "Nominal value of each share",
        "all_time_high": "Highest price ever recorded",
        "all_time_low": "Lowest price ever recorded",
        "avg_volume_20d": "Average daily volume over 20 days",
        "avg_delivery_20d": "Average delivery percentage over 20 days",
        "book_value_per_share": "Book value divided by total shares",
        "dividend_yield": "Annual dividend as percentage of price",
        "ttm_eps": "Trailing twelve months earnings per share",
        "ttm_pe": "Trailing twelve months price to earnings ratio",
        "pb_ratio": "Price to book value ratio",
        "sector_pe": "Average PE ratio for the sector",
        "pe_ratio": "Price to earnings ratio",
    }
    return descriptions.get(key, key)

def _extract_current_price(soup: BeautifulSoup) -> Optional[Dict]:
    candidates = [
        "#Nse_Prc_tick",
        "#Bse_Prc_tick",
        "span#Nse_Live_Price",
        "span#Bse_Live_Price",
        "div.inprice1.nsecp",
        "div.pcnsb",
        "span.stprbook",
        "div#stickystock span#ltpid",
    ]
    for sel in candidates:
        el = soup.select_one(sel)
        if el:
            raw = el.get_text(strip=True)
            val = _to_number(raw)
            if val is not None:
                return {"value": val, "unit": "₹", "description": _metric_desc("current_price"), "raw_value": raw}
    return None

def _extract_table_metrics(soup: BeautifulSoup) -> Dict[str, Dict]:
    metrics: Dict[str, Dict] = {}
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) != 2:
                continue
            label = tds[0].get_text(" ", strip=True)
            value_raw = tds[1].get_text(" ", strip=True)
            key = _map_metric(label)
            if not key or not value_raw:
                continue
            unit = _detect_unit(value_raw)
            val = _to_number(value_raw)
            metrics[key] = {"value": val, "unit": unit, "description": _metric_desc(key), "raw_value": value_raw}
    return metrics

def _extract_inline_metrics(soup: BeautifulSoup) -> Dict[str, Dict]:
    metrics: Dict[str, Dict] = {}
    for row in soup.select("li, div"):
        lbl = row.find(["span", "div"], string=True)
        if not lbl:
            continue
        label = lbl.get_text(" ", strip=True)
        key = _map_metric(label or "")
        if not key:
            continue
        val_el = lbl.find_next_sibling(["span", "div"])
        if not val_el:
            continue
        value_raw = val_el.get_text(" ", strip=True)
        if not value_raw:
            continue
        unit = _detect_unit(value_raw)
        val = _to_number(value_raw)
        metrics[key] = {"value": val, "unit": unit, "description": _metric_desc(key), "raw_value": value_raw}
    return metrics

def get_stock_data(company_name: str) -> Dict:
    """
    Resolve company via autosuggest and scrape key stock metrics from the overview page.
    Returns a dict; returns {} on failure.
    """
    if not company_name:
        return {}
    try:
        base_url = find_company_url(company_name)
        if not base_url:
            return {}
        with _client() as c:
            r = c.get(base_url)
            if r.status_code != 200 or "Access Denied" in r.text:
                return {}
            soup = BeautifulSoup(r.text, "lxml")

        metrics: Dict[str, Dict] = {}
        cp = _extract_current_price(soup)
        if cp:
            metrics["current_price"] = cp

        metrics.update(_extract_table_metrics(soup))

        inline = _extract_inline_metrics(soup)
        for k, v in inline.items():
            if k not in metrics:
                metrics[k] = v

        if "pe_ratio" not in metrics and "ttm_pe" in metrics:
            metrics["pe_ratio"] = {**metrics["ttm_pe"], "description": _metric_desc("pe_ratio")}

        return {
            "company_name": company_name,
            "source_url": base_url,
            "as_of": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }
    except Exception:
        return {}

if __name__ == "__main__":
    import sys, json
    name = " ".join(sys.argv[1:]).strip() or "Reliance Industries"
    print(json.dumps(scrape_company_stock(name), ensure_ascii=False, indent=2))

