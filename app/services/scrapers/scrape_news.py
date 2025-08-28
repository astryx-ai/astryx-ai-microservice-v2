"""
News scraper for Moneycontrol.
"""
import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime, timedelta

import httpx
from bs4 import BeautifulSoup

# User agent pool to avoid detection
UA_POOL = [
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

BASE_HEADERS = {
	"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
	"Accept-Language": "en-US,en;q=0.9",
	"Connection": "keep-alive",
	"Referer": "https://www.moneycontrol.com/",
}


def client(headers: Optional[dict] = None) -> httpx.Client:
	h = BASE_HEADERS.copy()
	h["User-Agent"] = random.choice(UA_POOL)
	if headers:
		h.update(headers)
	return httpx.Client(http2=True, headers=h, timeout=20, follow_redirects=True)


def get_html(c: httpx.Client, url: str, retries: int = 2) -> Optional[str]:
	for _ in range(max(1, retries)):
		try:
			r = c.get(url)
			if r.status_code == 200 and "Access Denied" not in r.text:
				return r.text
		except Exception:
			pass
		c.headers["User-Agent"] = random.choice(UA_POOL)  # rotate UA
	return None


def normalize_date(date_text: Optional[str]) -> Optional[str]:
	if not date_text:
		return None
	t = date_text.strip()
	if re.match(r"\d{4}-\d{2}-\d{2}", t):
		return t.split("T")[0]
	m = re.search(r"(\d{1,2})\s*([A-Za-z]+)\s*,?\s*(\d{4})", t)
	if m:
		d, mon, y = m.groups()
		mm = {
			"Jan": "01","Feb": "02","Mar": "03","Apr": "04","May": "05","Jun": "06",
			"Jul": "07","Aug": "08","Sep": "09","Oct": "10","Nov": "11","Dec": "12"
		}.get(mon[:3], "01")
		return f"{y}-{mm}-{d.zfill(2)}"
	return t


def extract_list_from_jsonld(html: str) -> List[Dict]:
	soup = BeautifulSoup(html, "lxml")
	items: List[Dict] = []
	for s in soup.find_all("script", {"type": "application/ld+json"}):
		try:
			data = json.loads(s.string or "")
		except Exception:
			continue
		objs = data if isinstance(data, list) else [data]
		for obj in objs:
			if isinstance(obj, dict) and isinstance(obj.get("itemListElement"), list):
				for it in obj["itemListElement"]:
					url = (it or {}).get("url")
					name = (it or {}).get("name")
					if url and name:
						items.append({"title": name.strip(), "url": url.strip(), "date": None})
		if items:
			break
	return items


def extract_article(html: str) -> Tuple[str, Optional[str]]:
	soup = BeautifulSoup(html, "lxml")
	# content
	parts: List[str] = []
	selectors = [
		("article", "p"),
		(".content_wrapper", "p"),
		(".article-desc", "p"),
		(".article_content", "p"),
		("#article-main", "p"),
		(None, "p"),
	]
	for scope_sel, p_sel in selectors:
		scope = soup.select_one(scope_sel) if scope_sel else soup
		if not scope:
			continue
		for p in scope.select(p_sel):
			t = (p.get_text(" ", strip=True) or "").strip()
			if t and not re.search(r"Copyright|Disclaimer", t, re.I):
				parts.append(t)
		if parts:
			break
	# date
	meta = soup.select_one("meta[property='article:published_time']")
	date_text = meta["content"].strip() if meta and meta.get("content") else None
	if not date_text:
		for sel in [".article_schedule", ".date", "time", "meta[name='publishdate']"]:
			el = soup.select_one(sel)
			if el:
				date_text = el.get("content", None) or el.get_text(strip=True)
				break
	return " ".join(parts).strip(), normalize_date(date_text)


def find_company_url(company_name: str) -> Optional[str]:
	"""Resolve Moneycontrol company overview URL using autosuggestion API.
	Prefer strong token-overlap matches to avoid wrong tag pages.
	"""
	api = "https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php"
	params = {"classic": "true", "query": company_name, "type": 1, "format": "json"}
	try:
		with client() as c:
			r = c.get(api, params=params)
			if r.status_code != 200:
				return None
			data = r.json()
	except Exception:
		return None
	if not isinstance(data, list) or not data:
		return None

	def norm_tokens(s: str):
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
			score += 0.2
		if score > best_score:
			best_score = score
			best_link = link
	if best_link:
		return best_link
	return None


def derive_news_url_from_company_url(company_url: str) -> Optional[str]:
	"""
	Convert company overview URL like:
	  https://www.moneycontrol.com/india/stockpricequote/<sector>/<slug>/<code>
	to company news URL:
	  https://www.moneycontrol.com/company-article/<slug>/news/<code>
	"""
	try:
		parts = [p for p in urlparse(company_url).path.split("/") if p]
		if len(parts) >= 2:
			slug = parts[-2]
			code = parts[-1]
			return f"https://www.moneycontrol.com/company-article/{slug}/news/{code}"
	except Exception:
		return None
	return None


def fallback_tag_url(company_url: str, company_name: str) -> str:
	try:
		parts = [p for p in urlparse(company_url).path.split("/") if p]
		slug = parts[-2] if len(parts) >= 2 else parts[-1]
	except Exception:
		slug = re.sub(r"\s+", "-", company_name.strip().lower())
	slug = slug.lower()
	return f"https://www.moneycontrol.com/news/tags/{slug}.html"


def scrape_company_news(company_name: str, limit: int = 20) -> List[Dict]:
	"""
	Scrape the most recent company news and return a list of dicts:
	{title, date, url, content}. Returns [] on failure.
	"""
	if not company_name:
		return []

	cutoff_date = datetime.now() - timedelta(days=180)  # only keep last ~6 months
	try:
		base_url = find_company_url(company_name)
		if not base_url:
			return []

		news_url = derive_news_url_from_company_url(base_url) or fallback_tag_url(base_url, company_name)

		with client() as c:
			_ = get_html(c, "https://www.moneycontrol.com/")  # warm-up
			listing_html = get_html(c, news_url)
			if not listing_html:
				alt = fallback_tag_url(base_url, company_name)
				if alt != news_url:
					listing_html = get_html(c, alt)
			if not listing_html:
				return []

			items = extract_list_from_jsonld(listing_html)

			# Fallback: scan anchors if JSON-LD missing
			if not items:
				soup = BeautifulSoup(listing_html, "lxml")
				for a in soup.select("a"):
					href = a.get("href", "")
					title = a.get_text(strip=True)
					if href and title and "/news/" in href and href.startswith("http"):
						items.append({"title": title, "url": href, "date": None})
						if len(items) >= limit:
							break

			if not items:
				return []

			results: List[Dict] = []
			for it in items:
				url = it["url"]
				ah = get_html(c, url)
				if not ah:
					amp_url = url.rstrip("/") + "/amp"
					ah = get_html(c, amp_url)
				if not ah:
					continue

				content, date_str = extract_article(ah)

				# Parse date & filter out old stuff
				keep_article = True
				if date_str:
					try:
						art_date = datetime.strptime(date_str, "%Y-%m-%d")
						if art_date < cutoff_date:
							keep_article = False
					except:
						pass

				if keep_article:
					results.append({
						"title": it.get("title") or "",
						"date": date_str,
						"url": url,
						"content": content
					})

			# Sort newest first
			results.sort(key=lambda x: x.get("date") or "", reverse=True)

			return results[:limit]

	except Exception:
		return []


# Backward-compat alias (fixes the import in fetch_news.py)
def get_news(company_name: str, limit: int = 10) -> List[Dict]:
	return scrape_company_news(company_name, limit)


def build_tag_base(company_name: str) -> str:
	slug = re.sub(r"\s+", "-", (company_name or "").strip().lower())
	return f"https://www.moneycontrol.com/news/tags/{slug}.html"


def page_url(base: str, n: int) -> str:
	if n <= 1:
		return base
	return base.rstrip("/") + f"/page-{n}/"


def save_atomic(path: str, data: List[Dict]) -> None:
	tmp = path + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
	os.replace(tmp, path)


def scrape(company_name: str, pages: int = 3, delay: float = 1.2, out_dir: Optional[str] = None) -> str:
	base = build_tag_base(company_name)
	out_dir = out_dir or f"moneycontrol_{company_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	os.makedirs(out_dir, exist_ok=True)

	listings: List[Dict] = []
	results: List[Dict] = []
	listings_path = os.path.join(out_dir, "listings.json")
	results_path = os.path.join(out_dir, f"{company_name.lower().replace(' ', '_')}_news.json")

	with client() as c:
		_ = get_html(c, "https://www.moneycontrol.com/")
		try:
			# Listing pages
			for n in range(1, pages + 1):
				url = page_url(base, n)
				print(f"Listing {n}: {url}")
				html = get_html(c, url)
				if not html:
					print(f"Listing {n}: access denied or empty, stopping.")
					break
				items = extract_list_from_jsonld(html)
				if not items:
					print(f"Listing {n}: no JSON-LD ItemList found, stopping.")
					break
				listings.extend(items)
				save_atomic(listings_path, listings)
				print(f"Saved {len(listings)} listings -> {listings_path}")
				time.sleep(delay + random.uniform(0, 0.6))

			# Articles
			for i, art in enumerate(listings, 1):
				print(f"[{i}/{len(listings)}] {art['title'][:80]}...")
				ah = get_html(c, art["url"])
				if not ah:
					amp_url = art["url"].rstrip("/") + "/amp"
					ah = get_html(c, amp_url)
				if not ah:
					results.append({**art, "content": "", "date": None})
					save_atomic(results_path, results)
					continue

				content, date = extract_article(ah)
				results.append({
					"title": art["title"],
					"url": art["url"],
					"content": content,
					"date": date
				})
				save_atomic(results_path, results)
				print(f"Saved progress ({len(results)} articles) -> {results_path}")
				time.sleep(delay + random.uniform(0, 0.7))

		except KeyboardInterrupt:
			print("\nInterrupted. Writing partial data...")
			save_atomic(listings_path, listings)
			save_atomic(results_path, results)

	print(f"Total articles scraped: {len(results)}")
	print(f"Saved to: {results_path}")
	return results_path


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--company", type=str, required=True, help="Company/tag name to scrape news for")
	ap.add_argument("--pages", type=int, default=3, help="Number of listing pages to scrape")
	ap.add_argument("--delay", type=float, default=1.2, help="Delay between requests (seconds)")
	ap.add_argument("--out", type=str, default=None, help="Output directory")
	args = ap.parse_args()
	scrape(args.company, args.pages, args.delay, args.out)
