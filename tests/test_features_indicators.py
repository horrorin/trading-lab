import numpy as np
import pandas as pd

from trading_lab.features.indicators import rsi


def test_rsi_bounds_and_length():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    # Smooth upward series to avoid pathological zeros
    px = pd.Series(np.linspace(100, 150, 50), index=idx, name="PX")

    out = rsi(px, window=14)

    # RSI is bounded in [0, 100] (allowing NaNs already dropped)
    assert (out >= 0).all()
    assert (out <= 100).all()
    assert out.name == "PX_rsi_14"


def test_rsi_returns_series_without_all_nan():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    px = pd.Series([100.0] * 30, index=idx, name="PX")  # flat line

    out = rsi(px, window=14)

    # With a flat series, RSI may be NaN due to 0/0; our implementation drops NaNs.
    # Ensure function returns a Series (possibly empty) without raising.
    assert isinstance(out, pd.Series)
