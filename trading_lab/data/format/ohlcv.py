from __future__ import annotations

from typing import Sequence
import pandas as pd

DEFAULT_KEEP_COLS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def normalize_ohlcv(
    df: pd.DataFrame, keep_cols: Sequence[str] = DEFAULT_KEEP_COLS
) -> pd.DataFrame:
    """
    Normalize OHLCV:
    - DatetimeIndex
    - Sorted
    - Keep subset of expected OHLCV columns (if present)
    - Drop Yahoo extras (Dividends, Stock Splits, etc.)
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    cols = [c for c in keep_cols if c in out.columns]
    out = out[cols]

    return out


def merge_timeseries(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge existing + new OHLCV with deduplication on index.
    New data overwrites existing data on overlapping timestamps.
    """
    if existing is None or existing.empty:
        return new.copy()
    if new is None or new.empty:
        return existing.copy()

    # Concatenate then drop duplicates, keeping last (new wins)
    merged = pd.concat([existing, new], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    return merged
