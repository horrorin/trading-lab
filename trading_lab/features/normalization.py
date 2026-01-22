import pandas as pd


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling z-score normalization:
    z_t = (x_t - mean_t) / std_t
    """
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    z = (series - mu) / sd
    z.name = f"{series.name}_z_{window}"
    return z.dropna()


def clip_series(
    series: pd.Series, lower: float | None = None, upper: float | None = None
) -> pd.Series:
    """
    Convenience wrapper around pandas Series.clip with naming preservation.
    """
    out = series.clip(lower=lower, upper=upper)
    out.name = series.name
    return out
