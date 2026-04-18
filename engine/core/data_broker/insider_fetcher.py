"""SEC EDGAR Form 4 fetcher for insider transaction data.

Fetches open-market purchase transactions by officers and directors.
Excludes 10%-owner-only filers (their trades are often mechanical/hedging).

Rate limit: SEC allows 10 req/sec. We target ~8 req/sec with a 0.13s delay.
All callers should use fetch_insider_transactions() as the public entry point;
it is already wrapped by the InsiderFlow feature's cache layer.
"""

import time
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_HEADERS_DATA = {
    "User-Agent": "stock_bot research coolestarr@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
_HEADERS_WWW = {
    "User-Agent": "stock_bot research coolestarr@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}
_RATE_DELAY = 0.13   # seconds between requests

_cik_cache: Dict[str, Optional[str]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, headers: dict, **kw) -> requests.Response:
    time.sleep(_RATE_DELAY)
    r = requests.get(url, headers=headers, timeout=15, **kw)
    r.raise_for_status()
    return r


def _extract_text(element, path: str) -> Optional[str]:
    el = element.find(path)
    return el.text.strip() if el is not None and el.text else None


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------

def get_cik(ticker: str) -> Optional[str]:
    """Returns zero-padded 10-digit CIK string for ticker, or None."""
    ticker_upper = ticker.upper()
    if ticker_upper in _cik_cache:
        return _cik_cache[ticker_upper]

    try:
        data = _get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_HEADERS_WWW,
        ).json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                _cik_cache[ticker_upper] = cik
                return cik
    except Exception as e:
        logger.warning(f"CIK lookup failed for {ticker}: {e}")

    _cik_cache[ticker_upper] = None
    return None


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------

def _get_form4_list(cik: str, start: str, end: str) -> List[Dict]:
    """Returns [{accession_number, filing_date}] for Form 4/4A in date range."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        data = _get(url, headers=_HEADERS_DATA).json()
    except Exception as e:
        logger.warning(f"Submissions fetch failed for CIK {cik}: {e}")
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    results = []
    for i, form in enumerate(forms):
        if form in ("4", "4/A") and start <= dates[i] <= end:
            results.append({
                "accession_number": accessions[i],
                "filing_date":      dates[i],
            })

    # Handle paginated older filings
    for batch_meta in data.get("filings", {}).get("files", []):
        batch_name = batch_meta.get("name", "")
        if not batch_name:
            continue
        try:
            batch = _get(
                f"https://data.sec.gov/submissions/{batch_name}",
                headers=_HEADERS_DATA,
            ).json()
            b_forms      = batch.get("form", [])
            b_dates      = batch.get("filingDate", [])
            b_accessions = batch.get("accessionNumber", [])
            for i, form in enumerate(b_forms):
                if form in ("4", "4/A") and start <= b_dates[i] <= end:
                    results.append({
                        "accession_number": b_accessions[i],
                        "filing_date":      b_dates[i],
                    })
        except Exception as e:
            logger.warning(f"Batch fetch failed ({batch_name}): {e}")

    return results


# ---------------------------------------------------------------------------
# Form 4 XML parsing
# ---------------------------------------------------------------------------

def _parse_form4(cik: str, accession_number: str) -> List[Dict]:
    """Downloads and parses one Form 4 filing. Returns list of transaction dicts."""
    acc_clean = accession_number.replace("-", "")
    cik_int   = int(cik)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{acc_clean}.txt"

    try:
        content = _get(url, headers=_HEADERS_WWW).text
    except Exception as e:
        logger.debug(f"Filing fetch failed ({accession_number}): {e}")
        return []

    # Extract <ownershipDocument>…</ownershipDocument> from the SGML wrapper
    start_tag = "<ownershipDocument>"
    end_tag   = "</ownershipDocument>"
    xs = content.find(start_tag)
    xe = content.find(end_tag)
    if xs == -1 or xe == -1:
        return []
    xml_text = content[xs : xe + len(end_tag)]

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.debug(f"XML parse error ({accession_number}): {e}")
        return []

    # ---- Reporting owner info ----
    owner_name = ""
    is_officer = is_director = is_ten_pct = False
    officer_title = ""

    for owner in root.findall(".//reportingOwner"):
        owner_name = _extract_text(owner, ".//rptOwnerName") or owner_name
        rel = owner.find("reportingOwnerRelationship")
        if rel is not None:
            is_officer   = _extract_text(rel, "isOfficer")          == "1"
            is_director  = _extract_text(rel, "isDirector")         == "1"
            is_ten_pct   = _extract_text(rel, "isTenPercentOwner")  == "1"
            officer_title = _extract_text(rel, "officerTitle") or ""

    # Skip pure 10%-owner filers (mechanical hedges / buybacks, not informational)
    if is_ten_pct and not (is_officer or is_director):
        return []

    role = officer_title if is_officer else ("Director" if is_director else "Unknown")

    # ---- Non-derivative transactions ----
    transactions = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        txn_type = _extract_text(txn, ".//transactionCoding/transactionType")
        if txn_type != "P":   # open-market purchases only
            continue

        txn_date = _extract_text(txn, ".//transactionDate/value")
        if not txn_date:
            continue

        shares_str = _extract_text(txn, ".//transactionShares/value")
        price_str  = _extract_text(txn, ".//transactionPricePerShare/value")

        try:
            shares = float(shares_str) if shares_str else 0.0
            price  = float(price_str)  if price_str  else 0.0
        except ValueError:
            continue

        transactions.append({
            "transaction_date": txn_date,
            "transaction_type": "P",
            "shares":           shares,
            "price":            price,
            "insider_name":     owner_name,
            "insider_role":     role,
            "accession_number": accession_number,
        })

    return transactions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_insider_transactions(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch all open-market officer/director purchases for ticker in [start, end].

    Args:
        ticker: Equity ticker symbol.
        start:  ISO date string 'YYYY-MM-DD'.
        end:    ISO date string 'YYYY-MM-DD'.

    Returns:
        DataFrame with columns: ticker, filing_date, transaction_date,
        transaction_type, price, shares, insider_name, insider_role, accession_number.
    """
    cik = get_cik(ticker)
    if not cik:
        return pd.DataFrame()

    filing_list = _get_form4_list(cik, start, end)
    if not filing_list:
        return pd.DataFrame()

    all_txns = []
    for filing in filing_list:
        txns = _parse_form4(cik, filing["accession_number"])
        for t in txns:
            t["ticker"]       = ticker
            t["filing_date"]  = filing["filing_date"]
        all_txns.extend(txns)

    if not all_txns:
        return pd.DataFrame()

    df = pd.DataFrame(all_txns)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["filing_date"]      = pd.to_datetime(df["filing_date"])
    return df
