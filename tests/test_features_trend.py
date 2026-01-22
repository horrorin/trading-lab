import numpy as np
import pandas as pd

from trading_lab.features.trend import sma, ema, sma_crossover_signal


def test_sma_matches_pandas_rolling_mean():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    s = pd.Series([1, 2, 3, 4, 5, 6], index=idx, name="X").astype(float)

    out = sma(s, window=3)
    expected = s.rolling(3).mean().dropna()

    assert out.index.equals(expected.index)
    assert np.allclose(out.values, expected.values)
    assert out.name == "X_sma_3"


def test_ema_matches_pandas_ewm_mean():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    s = pd.Series([1, 2, 3, 4, 5, 6], index=idx, name="X").astype(float)

    out = ema(s, span=3)
    expected = s.ewm(span=3, adjust=False).mean().dropna()

    assert out.index.equals(expected.index)
    assert np.allclose(out.values, expected.values)
    assert out.name == "X_ema_3"


def test_sma_crossover_signal_outputs_pm_one():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    px = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=idx, name="PX").astype(float)

    sig = sma_crossover_signal(px, fast=2, slow=5)

    # Signal should only contain -1 or +1
    assert set(sig.unique()).issubset({-1, 1})
    assert sig.name == "PX_sma_x_2_5"
