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
    # ~80-stock proxy for Russell 2000 / S&P 600 small-to-mid-cap universe.
    # Survivors-only snapshot (same caveat as all universes here).
    # Skews liquid: min ~$500M market cap at time of selection so yfinance
    # has clean daily OHLCV. Sector-balanced across 8 groups.
    "SMALL_MID_CAP": [
        # Semiconductors / hardware
        "AMBA", "POWI", "AEIS", "FORM", "COHU", "DIOD", "SMTC", "SLAB", "ICHR",
        # Software / internet
        "EGHT", "BAND", "LPSN", "MGNI", "PUBM",
        # Healthcare devices / services
        "LMAT", "NEOG", "PDCO", "PAHC", "CRVL", "IRTC",
        # Biotech (established, multi-year history)
        "ACAD", "NVCR", "HALO", "INVA", "NVAX",
        # Regional banks
        "IBOC", "WSFS", "HAFC", "GBCI", "BANF", "WAFD", "TRMK", "NBTB", "CATY",
        "SFNC", "UMBF", "FFIN",
        # Specialty finance
        "WRLD", "PRAA", "FCNCA",
        # Restaurants / consumer discretionary
        "BJRI", "CAKE", "JACK", "DINE", "PLAY", "RICK", "CATO", "BOOT",
        # Industrials / business services
        "AIN", "GMS", "ARCB", "MRCY", "HURN", "PLUS", "ASTE", "KFRC",
        # Energy (small E&P / oilfield services)
        "CIVI", "RES", "VNOM", "MTDR",
        # REITs
        "IIPR", "GOOD", "ROIC", "NHI", "STAG",
        # Materials / building products
        "MLI", "HWKN", "TREX", "UFP",
    ],
    "MEGA_CAP_TECH": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    ],
    "DOW_30": [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
        "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
    ],
    "SECTOR_ETFS": [
        "XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"
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
    "TOP_200": [
        # Mega cap
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "LLY",
        "JPM", "V", "MA", "UNH", "XOM", "COST", "HD", "PG", "JNJ", "ABBV",
        "WMT", "NFLX", "BAC", "CRM", "ORCL", "CVX", "MRK", "KO", "PEP", "AMD",
        "TMO", "LIN", "ADBE", "ACN", "MCD", "CSCO", "ABT", "WFC", "PM", "GE",
        "IBM", "NOW", "TXN", "QCOM", "ISRG", "CAT", "INTU", "AMGN", "GS", "MS",
        "VZ", "DHR", "AMAT", "NEE", "T", "BKNG", "AXP", "PFE", "BLK", "SPGI",
        "LOW", "RTX", "HON", "UNP", "SYK", "SCHW", "DE", "LRCX", "ELV", "PLD",
        "BMY", "GILD", "VRTX", "REGN", "MDT", "PANW", "CB", "ADI", "KLAC", "ADP",
        "CI", "BSX", "SBUX", "MDLZ", "TMUS", "MMC", "FI", "SO", "DUK", "CME",
        "CL", "ICE", "MU", "PYPL", "SNPS", "CDNS", "TJX", "EQIX", "MCO", "PH",
        # Large cap
        "APD", "AON", "EMR", "ITW", "NOC", "GD", "SLB", "EOG", "MPC", "PSX",
        "ORLY", "PNC", "USB", "TFC", "COF", "FDX", "CTAS", "ECL", "CMG", "NEM",
        "SHW", "HUM", "AFL", "CARR", "TRV", "AEP", "D", "SRE", "EXC", "XEL",
        "WM", "PCAR", "HCA", "DXCM", "MRNA", "ILMN", "KMB", "GIS", "SYY", "STZ",
        "ROP", "IDXX", "MSCI", "FAST", "AZO", "MCHP", "ON", "FTNT", "NXPI", "CRWD",
        "CPRT", "A", "GEHC", "ODFL", "AJG", "BKR", "DVN", "HAL", "FANG", "OXY",
        "KDP", "YUM", "MNST", "DG", "DLTR", "ROST", "TGT", "LULU", "NKE", "LVS",
        "AIG", "MET", "PRU", "ALL", "COIN", "GM", "F", "DAL", "UAL", "LUV",
        "AMT", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB", "EQR", "MAA",
        "PLTR", "SNOW", "DDOG", "ZS", "NET", "MDB", "SHOP", "SQ", "TTD", "MRVL",
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
