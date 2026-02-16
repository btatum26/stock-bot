import pandas as pd
import requests
from tqdm import tqdm
import time
from .engine import TradingEngine

class DataSnapshot:
    def __init__(self, engine: TradingEngine):
        self.engine = engine

    def get_top_1000_tickers(self):
        """
        Fetches a list of tickers, prioritizing a local tickers.txt file.
        """
        local_file = "data/tickers.txt"
        if os.path.exists(local_file):
            print(f"Loading tickers from local file: {local_file}")
            with open(local_file, "r") as f:
                return [line.strip() for line in f if line.strip()]

        tickers = set()
        sources = [
            # S&P 500
            "https://raw.githubusercontent.com/datasets/s-p-500-companies/master/data/constituents.csv",
            # S&P 400 (MidCap)
            "https://raw.githubusercontent.com/mfs-data/sp400-constituents/main/sp400.csv",
            # NASDAQ 100
            "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed.csv"
        ]

        print("Fetching ticker lists...")
        for url in sources:
            try:
                df = pd.read_csv(url)
                # Handle different column names across datasets
                symbol_col = next((c for c in df.columns if 'Symbol' in c or 'Ticker' in c), None)
                if symbol_col:
                    tickers.update(df[symbol_col].dropna().astype(str).tolist())
            except Exception as e:
                print(f"Warning: Could not fetch from {url}: {e}")

        # Clean tickers (remove duplicates, handle weird characters like '.' for '-' in yfinance)
        cleaned_tickers = []
        for t in tickers:
            t = str(t).strip().replace('.', '-')
            if t and len(t) <= 5 and t.isalpha():
                cleaned_tickers.append(t)

        final_list = sorted(list(set(cleaned_tickers)))
        
        # If we have too many, take the top 1000 (proxied by sorting, usually alphabetic here)
        # Ideally, we'd sort by market cap, but that requires another API call.
        if len(final_list) > 1000:
            return final_list[:1000]
        
        if not final_list:
            print("No tickers found online. Using manual fallback list.")
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK-B", "JNJ", "V"]
            
        return final_list

    def run(self, interval="1d", period="10y"):
        tickers = self.get_top_1000_tickers()
        print(f"Starting decade snapshot for {len(tickers)} tickers...")
        
        # Use tqdm for the "graphic" progress bar
        pbar = tqdm(tickers, desc="Downloading Market Data", unit="ticker")
        
        for ticker in pbar:
            retries = 3
            success = False
            while retries > 0 and not success:
                try:
                    # engine.sync_data already handles saving to DB and basic error printing
                    self.engine.sync_data(ticker, interval, period=period)
                    success = True
                except Exception as e:
                    retries -= 1
                    if retries > 0:
                        time.sleep(2) # Brief wait before retry
                    else:
                        pbar.write(f"Failed to sync {ticker} after multiple attempts.")
            
            # Rate limiting: small pause to avoid being blocked by Yahoo
            time.sleep(0.1)

        print("\nSnapshot complete.")
