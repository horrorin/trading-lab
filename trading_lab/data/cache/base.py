from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class CacheProvider(ABC):
    """
    Cache backend for OHLCV time series keyed by (ticker, timeframe).
    """

    @abstractmethod
    def read(self, ticker: str, timeframe: str) -> pd.DataFrame | None:
        raise NotImplementedError

    @abstractmethod
    def write(self, ticker: str, timeframe: str, df: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def path_for(self, ticker: str, timeframe: str) -> Path:
        raise NotImplementedError
