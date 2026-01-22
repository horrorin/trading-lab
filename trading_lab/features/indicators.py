import pandas as pd


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) implemented without TA-Lib.

    RSI = 100 - 100 / (1 + RS)
    where RS = avg_gain / avg_loss (smoothed).
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = f"{prices.name}_rsi_{window}"
    return rsi.dropna()
