import pandas as pd


def zscore(series: pd.Series, window: int | None = None) -> pd.Series:
    """
    Z-score normalization (rolling or full sample).
    """
    if window:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
    else:
        mean = series.mean()
        std = series.std()

    z = (series - mean) / std
    z.name = f"{series.name}_zscore"
    return z.dropna()


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize extreme values.
    """
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)
