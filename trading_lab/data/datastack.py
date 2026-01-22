from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from trading_lab.data.types import validate_timeframe
from trading_lab.data.format.ohlcv import normalize_ohlcv, merge_timeseries
from trading_lab.data.providers.base import DataProvider
from trading_lab.data.providers.yahoo import YahooDataProvider
from trading_lab.data.cache.base import CacheProvider
from trading_lab.data.cache.parquet import ParquetCacheProvider


def _to_ts(x: str) -> pd.Timestamp:
    return pd.Timestamp(x)


def _missing_ranges(
    existing: pd.DataFrame | None, start: str, end: str
) -> list[tuple[str, str]]:
    """
    Determine missing [start, end] ranges given existing cached data.

    Returns a list of (start, end) ranges to fetch.
    """
    req_start, req_end = _to_ts(start), _to_ts(end)

    if existing is None or existing.empty:
        return [(start, end)]

    ex_start = existing.index.min()
    ex_end = existing.index.max()

    ranges = []

    # Missing on the left
    if req_start < ex_start:
        ranges.append((str(req_start.date()), str((ex_start).date())))

    # Missing on the right
    if req_end > ex_end:
        ranges.append((str((ex_end).date()), str(req_end.date())))

    # Note: We ignore internal gaps for now (rare in daily). Can be added later.

    # Clean degenerate ranges (where start >= end)
    cleaned = []
    for a, b in ranges:
        if _to_ts(a) < _to_ts(b):
            cleaned.append((a, b))
    return cleaned


@dataclass
class DataStack:
    provider: DataProvider
    cache: CacheProvider

    def get_ohlcv(
        self,
        tickers: str | Sequence[str],
        start: str,
        end: str,
        timeframe: str = "1d",
        verbose: bool = True,
        **provider_kwargs,
    ) -> tuple[pd.DataFrame, ...]:
        """
        Get OHLCV for 1 or N tickers, using cache and extending it if needed.
        One parquet per (ticker, timeframe).
        """
        tf = validate_timeframe(timeframe)
        tlist = [tickers] if isinstance(tickers, str) else list(tickers)

        out = []

        for t in tlist:
            cached = self.cache.read(t, tf)
            if cached is not None and not cached.empty:
                cached = normalize_ohlcv(cached)

            needed = _missing_ranges(cached, start, end)

            if verbose:
                cache_path = self.cache.path_for(t, tf)
                if cached is None or cached.empty:
                    print(
                        f"[CACHE] MISS {t} {tf} -> will fetch {needed} ({cache_path.name})"
                    )
                else:
                    print(
                        f"[CACHE] HIT  {t} {tf} [{cached.index.min()} → {cached.index.max()}] -> need {needed}"
                    )

            # Fetch only missing segments
            merged = cached
            for seg_start, seg_end in needed:
                fetched = self.provider.fetch_ohlcv(
                    tickers=t,
                    start=seg_start,
                    end=seg_end,
                    timeframe=tf,
                    **provider_kwargs,
                )
                df_new = fetched.get(t)
                if df_new is None or df_new.empty:
                    continue
                merged = merge_timeseries(merged, normalize_ohlcv(df_new))

            if merged is None or merged.empty:
                raise RuntimeError(
                    f"No data available for {t} ({tf}) in {start}..{end}"
                )

            # Persist updated cache if we fetched anything
            if needed:
                self.cache.write(t, tf, merged)

            # Return requested slice
            out.append(merged.loc[start:end].copy())

        return tuple(out)

    def get_price_series(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Retrieve OHLCV price history for a given ticker.
        Uses provider + cache automatically.
        """
        return self.provider.fetch_ohlcv(
            ticker=ticker,
            start=start,
            end=end,
            timeframe=self.timeframe,
            cache=self.cache,
        )


class DataStackBuilder:
    """
    Builder to compose a DataStack with:
    - data provider (Yahoo default)
    - cache provider (Parquet default)
    """

    def __init__(self):
        self._provider: DataProvider = YahooDataProvider()
        self._cache: CacheProvider | None = None

    def with_provider(self, provider: DataProvider) -> "DataStackBuilder":
        self._provider = provider
        return self

    def with_parquet_cache(self, root_dir: str | Path) -> "DataStackBuilder":
        self._cache = ParquetCacheProvider(Path(root_dir))
        return self

    def build(self) -> DataStack:
        if self._cache is None:
            # Default cache directory
            self._cache = ParquetCacheProvider(Path("data/cache"))
        return DataStack(provider=self._provider, cache=self._cache)
