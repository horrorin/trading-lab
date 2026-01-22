import pandas as pd


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Simple Moving Average (SMA).
    """
    ma = series.rolling(window).mean()
    ma.name = f"{series.name}_sma_{window}"
    return ma.dropna()


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """
    Exponential Moving Average (EMA).
    """
    ma = series.ewm(span=span, adjust=False).mean()
    ma.name = f"{series.name}_ema_{span}"
    return ma.dropna()


def sma_crossover_signal(
    prices: pd.Series, fast: int = 20, slow: int = 100
) -> pd.Series:
    """
    Trend-following crossover signal:
    +1 if fast SMA > slow SMA else -1.

    Returns a {-1, +1} signal.
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()

    sig = (fast_ma > slow_ma).astype(int).replace({0: -1})
    sig.name = f"{prices.name}_sma_x_{fast}_{slow}"
    return sig.dropna()
