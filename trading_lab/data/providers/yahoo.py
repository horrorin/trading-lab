from __future__ import annotations

from typing import Sequence
import pandas as pd
import yfinance as yf

from trading_lab.data.providers import DataProvider
from trading_lab.data.format.ohlcv import normalize_ohlcv


def _as_list(x: str | Sequence[str]) -> list[str]:
    return [x] if isinstance(x, str) else list(x)


def _split_download_result(
    download_df: pd.DataFrame, tickers: list[str]
) -> dict[str, pd.DataFrame]:
    """
    Split yfinance.download output into per-ticker DataFrames.
    Works with MultiIndex or flat columns.
    """
    out: dict[str, pd.DataFrame] = {}

    if isinstance(download_df.columns, pd.MultiIndex):
        lvl0 = set(map(str, download_df.columns.get_level_values(0)))
        lvl1 = set(map(str, download_df.columns.get_level_values(1)))

        # (Ticker, Field) or (Field, Ticker)
        if any(t in lvl0 for t in tickers) and not any(t in lvl1 for t in tickers):
            for t in tickers:
                if t in lvl0:
                    out[t] = download_df[t].copy()
        else:
            for t in tickers:
                try:
                    out[t] = download_df.xs(t, axis=1, level=1).copy()
                except KeyError:
                    continue
    else:
        out[tickers[0]] = download_df.copy()

    return out


class YahooDataProvider(DataProvider):
    """
    Yahoo Finance provider via yfinance.download.
    """

    def fetch_ohlcv(
        self,
        tickers: str | Sequence[str],
        start: str,
        end: str,
        timeframe: str,
        auto_adjust: bool = False,
        repair: bool = True,
        progress: bool = False,
        group_by: str = "column",
    ) -> dict[str, pd.DataFrame]:
        tlist = _as_list(tickers)

        dl = yf.download(
            tickers=tlist,
            start=start,
            end=end,
            interval=timeframe,
            auto_adjust=auto_adjust,
            repair=repair,
            progress=progress,
            group_by=group_by,
        )

        if dl is None or dl.empty:
            # Return empty dict rather than raising: caller will decide what to do
            return {}

        split = _split_download_result(dl, tlist)

        out: dict[str, pd.DataFrame] = {}
        for t, df_t in split.items():
            if df_t is None or df_t.empty:
                continue
            out[t] = normalize_ohlcv(df_t)

        return out
