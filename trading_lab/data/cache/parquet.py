from __future__ import annotations

from pathlib import Path
import pandas as pd

from trading_lab.data.cache.base import CacheProvider


def _sanitize_ticker(ticker: str) -> str:
    return ticker.replace("^", "").replace("/", "_").replace("=", "_")


class ParquetCacheProvider(CacheProvider):
    """
    Parquet cache with one file per (ticker, timeframe):
      data/cache/{TICKER}_{TIMEFRAME}.parquet
    """

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, ticker: str, timeframe: str) -> Path:
        fname = f"{_sanitize_ticker(ticker)}_{timeframe}.parquet"
        return self.root_dir / fname

    def read(self, ticker: str, timeframe: str) -> pd.DataFrame | None:
        path = self.path_for(ticker, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def write(self, ticker: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self.path_for(ticker, timeframe)
        df.to_parquet(path)
