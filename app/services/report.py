from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import requests
import logging
from xml.etree import ElementTree as ET
from urllib.parse import urlparse, urljoin
import os

logger = logging.getLogger("ingestor.service")
# Import ticker repo to enable resolving/looking up tickers from Supabase (not wired yet)
from app.services.db import companies as ticker_repo  # noqa: F401

class IngestorService:
    """Service for ingesting and extracting corporate data from various sources."""

    def __init__(self):
        """Initialize the ingestor service."""
        pass

    def extract_shareholding_pattern(
        self, 
        instrument_key: Optional[str] = None, 
        scripcode: Optional[str] = None,
        from_year: int = 0,
        to_year: int = 0,
        max_xbrl_files: int = 3,
        stock_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract shareholding pattern data for a given instrument from BSE API.
        Save discovered XBRL/XML files under repo-local tmp/ only if the detected year(s) fall within [from_year, to_year].
        If scripcode is not provided, optionally resolve it from a stock/company name query.
        """
        try:
            # Resolve token if needed

            if not scripcode and stock_query:
                scripcode = self._resolve_scripcode_from_query(stock_query)

            if not scripcode:
                logger.warning("No scripcode provided and resolution from query failed")
                return self._get_error_response(instrument_key or "", "No scripcode provided")
            if not from_year or not to_year or from_year > to_year:
                return self._get_error_response(instrument_key or "", "fromYear and toYear are required, and fromYear <= toYear")

            # Build default instrument_key if missing
            if not instrument_key:
                instrument_key = f"BSE_EQ|SCRIPCODE_{scripcode}"

            url = f"https://api.bseindia.com/BseIndiaAPI/api/SHPQNewFormat/w?scripcode={scripcode}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Referer': 'https://www.bseindia.com/'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Normalize xbrlurl entries in Table to include full prefix
            self._normalize_table_xbrl_urls(data, base_prefix="https://www.bseindia.com/")

            # Detect years available in response
            detected_years = self._extract_years(data)

            # Filter Table rows to only include those within the requested year range
            original_table_count = 0
            filtered_table: Optional[List[Dict[str, Any]]] = None
            table_obj = data.get('Table')
            if isinstance(table_obj, list):
                original_table_count = len(table_obj)
                filtered_table = []
                for row in table_obj:
                    if isinstance(row, dict) and self._row_in_year_range(row, from_year, to_year):
                        filtered_table.append(row)

            # Build candidate links from filtered Table rows (prefer Table)
            filtered_links: List[str] = []
            if filtered_table:
                for row in filtered_table:
                    link = row.get('xbrlurl') or row.get('xbrlURL') or row.get('XBRLURL')
                    if isinstance(link, str) and link:
                        filtered_links.append(link)
                # dedupe while preserving order
                filtered_seen = set()
                filtered_links = [l for l in filtered_links if not (l in filtered_seen or filtered_seen.add(l))]

            # Fallback: if no Table-based links in range, consider regex-discovered links only if any detected_year in range
            if not filtered_links:
                if any(from_year <= y <= to_year for y in detected_years):
                    filtered_links = self._extract_xbrl_links(data)
                else:
                    return {
                        "instrument_key": instrument_key,
                        "scripcode": scripcode,
                        "requested_year_range": f"{from_year}-{to_year}",
                        "detected_years": sorted(detected_years),
                        "data_source": "bse_api",
                        "raw_response": data,
                        "status": "no_match"
                    }

            # Save XBRL files under repo-local tmp/
            stored_dir_abs: Optional[str] = None
            stored_files_info: List[Dict[str, Any]] = []
            if filtered_links:
                session = requests.Session()
                session.headers.update(headers)
                stored_dir_abs, stored_files_info = self._download_and_store_xbrl_files(
                    session=session,
                    links=filtered_links[: max(0, int(max_xbrl_files))],
                    scripcode=scripcode,
                    base_dir=os.path.join(self._get_project_root(), 'tmp'),
                    kind_tag="SHP"
                )

            # Prepare filtered raw response (replace Table with filtered rows if available)
            filtered_response = dict(data)
            if filtered_table is not None:
                filtered_response['Table'] = filtered_table

            return {
                "instrument_key": instrument_key,
                "scripcode": scripcode,
                "timeFrame": f"{from_year}-{to_year}",
                "fromYear": from_year,
                "toYear": to_year,
                "data_source": "bse_api",
                "raw_response": filtered_response,
                "raw_table_count": original_table_count,
                "filtered_table_count": len(filtered_table) if filtered_table is not None else None,
                "xbrl_links": filtered_links,
                "stored_files_dir": stored_dir_abs,
                "stored_files": stored_files_info,
                "stored_files_count": len(stored_files_info),
                "extracted_at": datetime.now().isoformat(),
                "status": "success"
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {instrument_key}: {str(e)}")
            return self._get_error_response(instrument_key or "", f"API request failed: {str(e)}")
        except Exception as e:
            logger.exception(f"Error extracting shareholding pattern for {instrument_key}: {str(e)}")
            return self._get_error_response(instrument_key or "", f"Extraction failed: {str(e)}")



    def extract_corporate_governance_range(self,
        instrument_key: Optional[str] = None,
        scripcode: Optional[str] = None,
        from_year: int = 0,
        to_year: int = 0,
        stock_query: Optional[str] = None,
        max_xbrl_files: int = 10
    ) -> Dict[str, Any]:
        """
        Discover and fetch all Corporate Governance XBRL files between from_year and to_year by scraping
        the quarter navigate pages for CG links. Saves files under repo-local tmp/.
        """
        try:
            # Resolve token if needed
            if not scripcode and stock_query:
                scripcode = self._resolve_scripcode_from_query(stock_query)
            if not scripcode:
                return self._get_error_response(instrument_key or "", "No scripcode provided")
            if not from_year or not to_year or from_year > to_year:
                return self._get_error_response(instrument_key or "", "fromYear and toYear are required, and fromYear <= toYear")
            if not instrument_key:
                instrument_key = f"BSE_EQ|SCRIPCODE_{scripcode}"

            # Use Shareholding API to get quarter listings and navigate URLs
            url = f"https://api.bseindia.com/BseIndiaAPI/api/AnnualReport_New/w?scripcode={scripcode}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Referer': 'https://www.bseindia.com/'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Track counts and build filtered rows like shareholding
            table_obj = data.get('Table') if isinstance(data, dict) else None
            original_table_count = len(table_obj) if isinstance(table_obj, list) else 0
            filtered_rows: List[Dict[str, Any]] = []
            if isinstance(table_obj, list):
                for row in table_obj:
                    if isinstance(row, dict) and self._row_in_year_range(row, from_year, to_year):
                        filtered_rows.append(row)

            # For each filtered row, fetch navigateurl page and extract CG links
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bseindia.com/', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 'Accept-Language': 'en-US,en;q=0.9'})
            cg_links: List[str] = []
            import re
            for row in filtered_rows:
                nav = row.get('navigateurl')
                if not isinstance(nav, str) or not nav.strip():
                    continue
                page_url = urljoin("https://www.bseindia.com/", nav.lstrip('/'))
                try:
                    html_resp = session.get(page_url, timeout=30)
                    html_resp.raise_for_status()
                    html = html_resp.text
                    # find CG XML links
                    pattern_abs = rf"https?://[^\s\"'<>]*/CGXBRLDataXML/{scripcode}_[0-9]+_CG\.xml"
                    pattern_rel = rf"/(?:XBRLFILES|xbrlfiles)/CGXBRLDataXML/{scripcode}_[0-9]+_CG\.xml"
                    # generic patterns (without scripcode constraint) as fallback
                    pattern_abs_any = r"https?://[^\s\"'<>]*/CGXBRLDataXML/[A-Za-z0-9_]+_CG\.xml"
                    pattern_rel_any = r"/(?:XBRLFILES|xbrlfiles)/CGXBRLDataXML/[A-Za-z0-9_]+_CG\.xml"
                    found_abs = re.findall(pattern_abs, html, re.IGNORECASE)
                    found_rel = re.findall(pattern_rel, html, re.IGNORECASE)
                    found_abs_any = re.findall(pattern_abs_any, html, re.IGNORECASE)
                    found_rel_any = re.findall(pattern_rel_any, html, re.IGNORECASE)

                    candidates: List[str] = []
                    candidates.extend(found_abs)
                    candidates.extend(urljoin("https://www.bseindia.com/", l) for l in found_rel)
                    candidates.extend(found_abs_any)
                    candidates.extend(urljoin("https://www.bseindia.com/", l) for l in found_rel_any)

                    # Keep those that include expected scripcode in basename
                    for link in candidates:
                        try:
                            fname = os.path.basename(urlparse(link).path)
                            if fname.startswith(f"{scripcode}_") and fname.endswith("_CG.xml"):
                                cg_links.append(link)
                        except Exception:
                            continue

                    # Fallback: also scan the SEO URL page if present
                    seo = row.get('seourl')
                    if isinstance(seo, str) and seo.strip():
                        seo_url = urljoin("https://www.bseindia.com/", seo.lstrip('/'))
                        try:
                            seo_resp = session.get(seo_url, timeout=30)
                            if seo_resp.ok:
                                html2 = seo_resp.text
                                found_abs2 = re.findall(pattern_abs, html2, re.IGNORECASE)
                                found_rel2 = re.findall(pattern_rel, html2, re.IGNORECASE)
                                found_abs_any2 = re.findall(pattern_abs_any, html2, re.IGNORECASE)
                                found_rel_any2 = re.findall(pattern_rel_any, html2, re.IGNORECASE)
                                cand2: List[str] = []
                                cand2.extend(found_abs2)
                                cand2.extend(urljoin("https://www.bseindia.com/", l) for l in found_rel2)
                                cand2.extend(found_abs_any2)
                                cand2.extend(urljoin("https://www.bseindia.com/", l) for l in found_rel_any2)
                                for link in cand2:
                                    try:
                                        fname = os.path.basename(urlparse(link).path)
                                        if fname.startswith(f"{scripcode}_") and fname.endswith("_CG.xml"):
                                            cg_links.append(link)
                                    except Exception:
                                        continue
                        except Exception:
                            pass
                except Exception as ex:
                    logger.exception(f"Failed to scan navigateurl page {page_url}: {ex}")
                    continue

            # Dedupe and limit
            dedup = []
            seen = set()
            for link in cg_links:
                if link not in seen:
                    seen.add(link)
                    dedup.append(link)
            cg_links = dedup[: max(0, int(max_xbrl_files))]

            # Prepare filtered raw response mirroring shareholding
            filtered_response = dict(data) if isinstance(data, dict) else {}
            if isinstance(table_obj, list):
                filtered_response['Table'] = filtered_rows

            if not cg_links:
                # For parity with shareholding no_match, also include detected years and requested range
                detected_years = self._extract_years(data)
                return {
                    "instrument_key": instrument_key,
                    "scripcode": scripcode,
                    "timeFrame": f"{from_year}-{to_year}",
                    "fromYear": from_year,
                    "toYear": to_year,
                    "data_source": "bse_api",
                    "raw_response": filtered_response,
                    "raw_table_count": original_table_count,
                    "filtered_table_count": len(filtered_rows),
                    "xbrl_links": [],
                    "requested_year_range": f"{from_year}-{to_year}",
                    "detected_years": sorted(detected_years),
                    "stored_files_dir": None,
                    "stored_files": [],
                    "stored_files_count": 0,
                    "extracted_at": datetime.now().isoformat(),
                    "status": "no_match"
                }

            stored_dir_abs, stored_files_info = self._download_and_store_xbrl_files(
                session=session,
                links=cg_links,
                scripcode=scripcode,
                base_dir=os.path.join(self._get_project_root(), 'tmp'),
                kind_tag="CG"
            )

            return {
                "instrument_key": instrument_key,
                "scripcode": scripcode,
                "timeFrame": f"{from_year}-{to_year}",
                "fromYear": from_year,
                "toYear": to_year,
                "data_source": "bse_api",
                "raw_response": filtered_response,
                "raw_table_count": original_table_count,
                "filtered_table_count": len(filtered_rows),
                "xbrl_links": cg_links,
                "stored_files_dir": stored_dir_abs,
                "stored_files": stored_files_info,
                "stored_files_count": len(stored_files_info),
                "extracted_at": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            logger.exception(f"Error extracting CG range for {instrument_key}: {str(e)}")
            return self._get_error_response(instrument_key or "", f"CG range extraction failed: {str(e)}")


    # ----------------------- Helper/private methods and additional endpoints -----------------------
    def _extract_xbrl_links(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract XBRL links from the API response by scanning the JSON as text.
        """
        xbrl_links = []

        try:
            data_str = str(data)

            import re
            xbrl_patterns = [
                r"https?://[^\s\"'<>]+\.xbrl",
                r"https?://[^\s\"'<>]+/xbrl[^\s\"'<>]*",
                r"https?://[^\s\"'<>]*xbrl[^\s\"'<>]*",
            ]

            for pattern in xbrl_patterns:
                matches = re.findall(pattern, data_str, re.IGNORECASE)
                xbrl_links.extend(matches)

            xbrl_links = list(dict.fromkeys(xbrl_links))

        except Exception as e:
            logger.exception("Error extracting XBRL links")

        return xbrl_links

    def _normalize_table_xbrl_urls(self, data: Dict[str, Any], base_prefix: str) -> None:
        try:
            table = data.get('Table')
            if not isinstance(table, list):
                return
            for row in table:
                if not isinstance(row, dict):
                    continue
                key_candidates = ['xbrlurl', 'xbrlURL', 'XBRLURL']
                for key in key_candidates:
                    if key in row and isinstance(row[key], str) and row[key]:
                        url_val = row[key].strip()
                        parsed = urlparse(url_val)
                        if not parsed.scheme:
                            full_url = urljoin(base_prefix, url_val.lstrip('/'))
                            row[key] = full_url
        except Exception as exc:
            logger.exception("Failed normalizing Table xbrlurl")

    def _extract_years(self, obj: Any) -> List[int]:
        import re
        years = set()
        try:
            def walk(x: Any):
                if isinstance(x, dict):
                    for v in x.values():
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)
                else:
                    for y in re.findall(r'\b(19|20)\d{2}\b', str(x)):
                        if isinstance(y, tuple):
                            years.add(int(''.join(y)))
                        else:
                            years.add(int(y))
            walk(obj)
        except Exception as exc:
            logger.exception("Failed extracting years")
        return sorted(years)

    def _get_error_response(self, instrument_key: str, error_message: str) -> Dict[str, Any]:
        """
        Generate standardized error response.
        """
        return {
            "instrument_key": instrument_key,
            "data_source": "bse_api",
            "error": error_message,
            "xbrl_links": [],
            "extracted_at": datetime.now().isoformat(),
            "status": "error"
        }

    def _read_text_source(self, source: str, timeout_seconds: int = 30) -> str:
        if self._is_url(source):
            resp = requests.get(source, timeout=timeout_seconds)
            resp.raise_for_status()
            return resp.text
        if os.path.exists(source):
            with open(source, 'r', encoding='utf-8') as f:
                return f.read()
        return source

    def _is_url(self, value: str) -> bool:
        try:
            parsed = urlparse(value)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    def _compose_time_frame(self, quarter: Optional[str], year: Optional[int]) -> Optional[str]:
        if quarter and year:
            return f"{quarter} {year}"
        if year:
            return str(year)
        return None

    def _get_project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

    def _download_and_store_xbrl_files(
        self,
        session: Optional[requests.Session],
        links: List[str],
        scripcode: str,
        base_dir: Optional[str] = None,
        kind_tag: Optional[str] = None,
        name_prefix: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Download XBRL/XML/PDF files and store under repo-local tmp by default.
        Returns the absolute directory path and a list of stored file info.
        """
        dir_base = base_dir or os.path.join(self._get_project_root(), 'tmp')
        tag = (kind_tag or "GEN").upper()
        # Save in tmp/BSE/<scripcode>/<TAG>/ as requested
        target_dir = os.path.join(dir_base, 'BSE', scripcode, tag)
        os.makedirs(target_dir, exist_ok=True)

        stored: List[Dict[str, Any]] = []
        sess = session or requests.Session()
        for idx, link in enumerate(links):
            try:
                resp = sess.get(link, timeout=30)
                resp.raise_for_status()
                parsed = urlparse(link)
                name = os.path.basename(parsed.path) or f"file_{idx+1}.bin"
                if not os.path.splitext(name)[1]:
                    ctype = resp.headers.get('content-type', '').lower()
                    ext = '.pdf' if 'pdf' in ctype else ('.xml' if 'xml' in ctype else '.bin')
                    name = name + ext
                file_path = os.path.join(target_dir, name)
                with open(file_path, 'wb') as f:
                    f.write(resp.content)
                stored.append({
                    "link": link,
                    "path": file_path,
                    "size_bytes": len(resp.content),
                    "content_type": resp.headers.get('content-type')
                })
            except Exception as exc:
                logger.error(f"Failed to download/store from {link}: {exc}")
                stored.append({
                    "link": link,
                    "error": str(exc)
                })
        return target_dir, stored

    def extract_annual_results(self,
        instrument_key: Optional[str] = None,
        scripcode: Optional[str] = None,
        from_year: int = 0,
        to_year: int = 0,
        stock_query: Optional[str] = None,
        max_files: int = 25
    ) -> Dict[str, Any]:
        """
        Fetch Annual Report listings from BSE and download PDFDownload links within the year range.
        Stores under repo-local tmp/ as tmp/bse_pdf_{scripcode}_AR_{timestamp}/.
        API: https://api.bseindia.com/BseIndiaAPI/api/AnnualReport_New/w?scripcode={scripcode}
        """
        try:
            # Resolve scripcode from query if needed
            if not scripcode and stock_query:
                scripcode = self._resolve_scripcode_from_query(stock_query)
            if not scripcode:
                return self._get_error_response(instrument_key or "", "No scripcode provided")
            if not from_year or not to_year or from_year > to_year:
                return self._get_error_response(instrument_key or "", "fromYear and toYear are required, and fromYear <= toYear")
            if not instrument_key:
                instrument_key = f"BSE_EQ|SCRIPCODE_{scripcode}"

            url = f"https://api.bseindia.com/BseIndiaAPI/api/AnnualReport_New/w?scripcode={scripcode}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Referer': 'https://www.bseindia.com/'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            table_obj = data.get('Table') if isinstance(data, dict) else None
            original_table_count = len(table_obj) if isinstance(table_obj, list) else 0

            # Filter rows by Year in range and collect PDF links
            filtered_rows: List[Dict[str, Any]] = []
            pdf_links: List[str] = []
            if isinstance(table_obj, list):
                for row in table_obj:
                    if not isinstance(row, dict):
                        continue
                    y_raw = row.get('Year')
                    try:
                        y_int = int(str(y_raw).strip()) if y_raw is not None else None
                    except Exception:
                        y_int = None
                    if y_int is None or not (from_year <= y_int <= to_year):
                        continue
                    filtered_rows.append(row)
                    link = row.get('PDFDownload')
                    if isinstance(link, str) and link.strip():
                        norm = link.strip().replace("\\", "")
                        pdf_links.append(norm)

            # Dedupe and cap
            seen = set()
            pdf_links = [l for l in pdf_links if not (l in seen or seen.add(l))][: max(0, int(max_files))]

            # Prepare filtered raw response
            filtered_response = dict(data) if isinstance(data, dict) else {}
            if isinstance(table_obj, list):
                filtered_response['Table'] = filtered_rows

            if not pdf_links:
                return {
                    "instrument_key": instrument_key,
                    "scripcode": scripcode,
                    "timeFrame": f"{from_year}-{to_year}",
                    "fromYear": from_year,
                    "toYear": to_year,
                    "data_source": "bse_api",
                    "raw_response": filtered_response,
                    "raw_table_count": original_table_count,
                    "filtered_table_count": len(filtered_rows),
                    "pdf_links": [],
                    "stored_files_dir": None,
                    "stored_files": [],
                    "stored_files_count": 0,
                    "extracted_at": datetime.now().isoformat(),
                    "status": "no_match"
                }

            # Download PDFs
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bseindia.com/', 'Accept': 'application/pdf,*/*;q=0.8'})
            stored_dir_abs, stored_files_info = self._download_and_store_xbrl_files(
                session=session,
                links=pdf_links,
                scripcode=scripcode,
                base_dir=os.path.join(self._get_project_root(), 'tmp'),
                kind_tag="AR",
                name_prefix="bse_pdf"
            )

            return {
                "instrument_key": instrument_key,
                "scripcode": scripcode,
                "timeFrame": f"{from_year}-{to_year}",
                "fromYear": from_year,
                "toYear": to_year,
                "data_source": "bse_api",
                "raw_response": filtered_response,
                "raw_table_count": original_table_count,
                "filtered_table_count": len(filtered_rows),
                "pdf_links": pdf_links,
                "stored_files_dir": stored_dir_abs,
                "stored_files": stored_files_info,
                "stored_files_count": len(stored_files_info),
                "extracted_at": datetime.now().isoformat(),
                "status": "success"
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Annual results fetch failed for {instrument_key}: {str(e)}")
            return self._get_error_response(instrument_key or "", f"Annual results fetch failed: {str(e)}")
        except Exception as e:
            logger.exception(f"Error extracting annual results for {instrument_key}: {str(e)}")
            return self._get_error_response(instrument_key or "", f"Annual results extraction failed: {str(e)}")

    def extract_annual_reports(
        self, 
        instrument_key: str,
        year: Optional[int] = None,
        report_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract annual report data for a given instrument.
        """
        default_sections = [
            'financial_highlights',
            'management_discussion_analysis', 
            'directors_report',
            'auditors_report',
            'financial_statements'
        ]

        sections_to_extract = report_sections or default_sections

        return {
            "instrument_key": instrument_key,
            "year": year,
            "data_source": "placeholder",
            "sections_extracted": sections_to_extract,
            "annual_report_data": {
                "financial_highlights": {},
                "management_discussion_analysis": "",
                "directors_report": "",
                "auditors_report": "",
                "financial_statements": {
                    "balance_sheet": {},
                    "profit_loss": {},
                    "cash_flow": {}
                }
            },
            "extracted_at": datetime.now().isoformat(),
            "status": "placeholder"
        }

    def get_available_data_sources(self) -> List[str]:
        """
        Get list of available data sources for extraction.
        """
        return [
            "bse_corporate_filings",
            "nse_corporate_announcements", 
            "sebi_edgar",
            "company_websites",
            "annual_report_pdfs"
        ]

    def validate_instrument_key(self, instrument_key: str) -> bool:
        """
        Validate if the instrument key format is correct.
        """
        if not instrument_key or "|" not in instrument_key:
            return False

        parts = instrument_key.split("|")
        if len(parts) != 2:
            return False

        exchange_segment, isin = parts
        valid_segments = ["NSE_EQ", "BSE_EQ", "NSE_FO", "BSE_FO", "MCX_FO"]

        return exchange_segment in valid_segments and len(isin) >= 10

    def _row_in_year_range(self, row: Dict[str, Any], from_year: int, to_year: int) -> bool:
        try:
            import re
            # 1) Prefer quarter year from 'qtr' (e.g., 'June 2025')
            qtr_val = row.get('qtr')
            if isinstance(qtr_val, str):
                m = re.search(r'(19|20)\d{2}', qtr_val)
                if m:
                    y = int(m.group(0))
                    return from_year <= y <= to_year
            # 2) Next, try filing date year
            filing_dt = row.get('filing_date_time') or row.get('revised_date_time')
            if isinstance(filing_dt, str):
                m = re.search(r'^(\d{4})', filing_dt)
                if m:
                    y = int(m.group(1))
                    return from_year <= y <= to_year
                # or any 4-digit year in the string
                m2 = re.search(r'(19|20)\d{2}', filing_dt)
                if m2:
                    y = int(m2.group(0))
                    return from_year <= y <= to_year
            # 3) Then, consider 'yr' like '2024 - 2025' but only if any boundary is in range
            yr_val = row.get('yr')
            if isinstance(yr_val, str):
                years = [int(p.strip()) for p in yr_val.split('-') if p.strip().isdigit()]
                if years and any(from_year <= y <= to_year for y in years):
                    # Do not allow this to include outside years when a narrower range was requested and qtr is missing
                    # If a single year range requested, require that year to be present
                    if from_year == to_year:
                        return from_year in years
                    return True
            # 4) Conservative fallback: scan the row and only accept if ALL found years fall within the range
            found_years = self._extract_years(row)
            if found_years:
                return min(found_years) >= from_year and max(found_years) <= to_year
            return False
        except Exception:
            return False

    def _resolve_scripcode_from_query(self, stock_query: Optional[str]) -> Optional[str]:
        """Resolve BSE scripcode from a free-text company/stock query via companies repo."""
        try:
            q = (stock_query or "").strip()
            if not q:
                return None
            # Try multiple identifiers best-effort; companies.resolve_bse_scripcode handles name search
            from app.services.db import companies as _companies

            code = _companies.resolve_bse_scripcode(company_name=q)
            return code
        except Exception as exc:
            logger.warning(f"Failed to resolve scripcode from query '{stock_query}': {exc}")
            return None
