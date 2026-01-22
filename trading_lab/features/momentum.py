import pandas as pd


def rate_of_change(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Rate of Change (ROC): (P_t / P_{t-window}) - 1
    """
    roc = prices / prices.shift(window) - 1.0
    roc.name = f"{prices.name}_roc_{window}"
    return roc.dropna()


def momentum(prices: pd.Series, window: int = 252) -> pd.Series:
    """
    Simple momentum proxy: same as ROC over a longer horizon by default.
    """
    return rate_of_change(prices, window=window)
