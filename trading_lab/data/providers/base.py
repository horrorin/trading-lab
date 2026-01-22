from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
import pandas as pd


class DataProvider(ABC):
    """
    Fetch raw OHLCV data from an external source.
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        tickers: str | Sequence[str],
        start: str,
        end: str,
        timeframe: str,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """
        Returns a dict: {ticker: DataFrame(OHLCV)} for the requested range.
        """
        raise NotImplementedError
