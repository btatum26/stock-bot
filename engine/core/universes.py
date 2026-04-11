"""Hardcoded ticker universes for multi-ticker training.

# TODO: Replace static presets with an auto-sync source. Options:
#   - Scrape Wikipedia for S&P 500 / Nasdaq-100 / Dow constituents
#   - Pull from IEX Cloud / Polygon / yfinance index components
#   - Maintain a scheduled CSV refresh job under data/universes/
# Current behavior is a frozen snapshot — survivors-only, so backtests
# on historical periods will have a subtle survivorship bias until this
# is replaced with a time-varying (point-in-time) source.
"""

from typing import Dict, List


UNIVERSES: Dict[str, List[str]] = {
    "MEGA_CAP_TECH": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    ],
    "DOW_30": [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
        "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
    ],
    "SECTOR_ETFS": [
        "XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    ],
    "LIQUID_ETFS": [
        "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "TLT", "GLD", "SLV", "USO",
    ],
    "SEMICONDUCTORS": [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC",
        "MRVL", "ADI", "NXPI", "ON", "MCHP",
    ],
    "FINANCIALS_LARGE": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
        "PNC", "TFC", "COF", "BK", "STT",
    ],
    "HEALTHCARE_PHARMA": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "VRTX", "REGN", "MRNA",
    ],
    "CONSUMER_STAPLES": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "GIS",
        "KMB", "SYY", "HSY", "K", "CAG",
    ],
    "ENERGY_MAJORS": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD",
        "DVN", "HES", "HAL", "BKR", "FANG",
    ],
    "REITS": [
        "PLD", "AMT", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB", "EQR",
        "VTR", "ARE", "MAA", "UDR", "ESS",
    ],
    "VOLATILITY_PRODUCTS": [
        "VXX", "UVXY", "SVXY", "VIXY",
    ],
    "FIXED_INCOME_ETFS": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "BND", "AGG", "TIP", "MUB",
        "EMB", "VCIT", "VCSH", "GOVT", "MBB",
    ],
    "INTERNATIONAL_ETFS": [
        "EFA", "EEM", "VEA", "VWO", "IEFA", "IEMG", "FXI", "EWJ", "EWZ", "EWG",
        "EWU", "INDA", "VGK", "MCHI", "EWT",
    ],
    "COMMODITIES": [
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "CORN", "WEAT", "CPER",
    ],
    "GROWTH_TECH": [
        "SNOW", "CRWD", "DDOG", "ZS", "NET", "MDB", "PANW", "SHOP", "SQ", "ROKU",
        "COIN", "PLTR", "AFRM", "U", "PATH",
    ],
    "DIVIDEND_ARISTOCRATS": [
        "JNJ", "PG", "KO", "PEP", "MMM", "ABT", "XOM", "CVX", "CL", "MCD",
        "WMT", "TGT", "ADP", "AFL", "EMR",
    ],
    "SMALL_CAP_GROWTH": [
        "IWO", "VBK", "SLYG", "IJT", "FYC",
    ],
    "LEVERAGED_ETFS": [
        "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "TNA", "TZA", "SOXL", "SOXS", "LABU",
    ],
}


def get_universe(name: str) -> List[str]:
    """Return the ticker list for a named universe.

    Raises:
        KeyError: If the universe name is not defined.
    """
    if name not in UNIVERSES:
        raise KeyError(
            f"Unknown universe '{name}'. Available: {list(UNIVERSES.keys())}"
        )
    return list(UNIVERSES[name])


def list_universes() -> List[str]:
    """Return sorted names of all defined universes."""
    return sorted(UNIVERSES.keys())
