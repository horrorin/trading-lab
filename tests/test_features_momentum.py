import numpy as np
import pandas as pd

from trading_lab.features.momentum import rate_of_change, momentum


def test_rate_of_change_definition():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    px = pd.Series([100, 101, 102, 103, 104, 105], index=idx, name="PX").astype(float)

    roc = rate_of_change(px, window=2)

    expected = (px / px.shift(2) - 1.0).dropna()
    assert roc.index.equals(expected.index)
    assert np.allclose(roc.values, expected.values)
    assert roc.name == "PX_roc_2"


def test_momentum_aliases_roc():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    px = pd.Series([100, 101, 102, 103, 104, 105], index=idx, name="PX").astype(float)

    m = momentum(px, window=3)
    expected = rate_of_change(px, window=3)

    assert m.index.equals(expected.index)
    assert np.allclose(m.values, expected.values)
