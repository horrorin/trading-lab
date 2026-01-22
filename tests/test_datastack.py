import pandas as pd
import pytest

from trading_lab.data.datastack import DataStackBuilder
from trading_lab.data.providers.base import DataProvider


def _mk_ohlcv(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(len(idx)),
            "High": range(len(idx)),
            "Low": range(len(idx)),
            "Close": range(len(idx)),
            "Adj Close": range(len(idx)),
            "Volume": [100] * len(idx),
        },
        index=idx,
    )
    return df


class DummyProvider(DataProvider):
    """
    Deterministic provider for tests.
    Records fetch calls and returns OHLCV for given ranges.
    """

    def __init__(self, data_by_ticker: dict[str, pd.DataFrame]):
        self.data_by_ticker = data_by_ticker
        self.calls: list[dict] = []

    def fetch_ohlcv(self, tickers, start, end, timeframe, **kwargs):
        # record call
        self.calls.append(
            {"tickers": tickers, "start": start, "end": end, "timeframe": timeframe}
        )

        # normalize tickers to list
        tlist = [tickers] if isinstance(tickers, str) else list(tickers)

        out = {}
        for t in tlist:
            df = self.data_by_ticker.get(t)
            if df is None or df.empty:
                continue
            # emulate provider: returns only requested slice
            out[t] = df.loc[start:end].copy()
        return out


@pytest.fixture
def dummy_provider():
    # Base dataset from 2000-01-01 to 2000-01-10
    base = _mk_ohlcv("2000-01-01", "2000-01-10")
    return DummyProvider({"^FCHI": base})


def test_stack_initial_fetch_writes_cache(tmp_path, dummy_provider):
    """
    First call: cache miss -> provider fetch -> parquet written -> slice returned.
    """
    ds = (
        DataStackBuilder()
        .with_provider(dummy_provider)
        .with_parquet_cache(tmp_path)
        .build()
    )

    (df,) = ds.get_ohlcv(
        "^FCHI", start="2000-01-01", end="2000-01-05", timeframe="1d", verbose=False
    )

    assert not df.empty
    assert df.index.min() == pd.Timestamp("2000-01-01")
    assert df.index.max() == pd.Timestamp("2000-01-05")

    # provider must have been called at least once
    assert len(dummy_provider.calls) >= 1

    # cache file must exist
    cache_file = tmp_path / "FCHI_1d.parquet"
    assert cache_file.exists()

    # cache should contain at least the requested range (likely more depending on implementation)
    cached = pd.read_parquet(cache_file)
    assert not cached.empty
    assert cached.index.min() <= pd.Timestamp("2000-01-01")


def test_stack_cache_hit_no_extra_fetch(tmp_path, dummy_provider):
    """
    Second call inside cached range should not re-fetch data.
    """
    ds = (
        DataStackBuilder()
        .with_provider(dummy_provider)
        .with_parquet_cache(tmp_path)
        .build()
    )

    # First call populates cache
    ds.get_ohlcv(
        "^FCHI", start="2000-01-01", end="2000-01-05", timeframe="1d", verbose=False
    )
    calls_after_first = len(dummy_provider.calls)

    # Second call is strictly inside already cached min/max -> no provider call expected
    ds.get_ohlcv(
        "^FCHI", start="2000-01-02", end="2000-01-04", timeframe="1d", verbose=False
    )
    calls_after_second = len(dummy_provider.calls)

    assert (
        calls_after_second == calls_after_first
    ), "Cache hit should not trigger provider fetch."


def test_stack_extends_cache_right_side(tmp_path):
    """
    If cache exists up to a date, requesting later end should fetch only the missing right segment and extend parquet.
    """
    # provider has longer data than initial request
    full = _mk_ohlcv("2000-01-01", "2000-01-20")
    provider = DummyProvider({"^FCHI": full})

    ds = DataStackBuilder().with_provider(provider).with_parquet_cache(tmp_path).build()

    # Populate cache with early range
    ds.get_ohlcv(
        "^FCHI", start="2000-01-01", end="2000-01-10", timeframe="1d", verbose=False
    )
    calls_after_first = len(provider.calls)

    # Now extend on the right
    (df2,) = ds.get_ohlcv(
        "^FCHI", start="2000-01-05", end="2000-01-15", timeframe="1d", verbose=False
    )
    calls_after_second = len(provider.calls)

    assert not df2.empty
    assert df2.index.min() == pd.Timestamp("2000-01-05")
    assert df2.index.max() == pd.Timestamp("2000-01-15")

    # Must have fetched additional data exactly once (for the right extension)
    assert calls_after_second == calls_after_first + 1

    # Verify the cache is extended
    cache_file = tmp_path / "FCHI_1d.parquet"
    cached = pd.read_parquet(cache_file)
    assert cached.index.max() >= pd.Timestamp("2000-01-15")


def test_stack_separate_cache_files_per_timeframe(tmp_path):
    """
    Same ticker but different timeframe should use different parquet files.
    """
    full = _mk_ohlcv("2000-01-01", "2000-01-10")
    provider = DummyProvider({"^FCHI": full})

    ds = DataStackBuilder().with_provider(provider).with_parquet_cache(tmp_path).build()

    ds.get_ohlcv(
        "^FCHI", start="2000-01-01", end="2000-01-05", timeframe="1d", verbose=False
    )
    ds.get_ohlcv(
        "^FCHI", start="2000-01-01", end="2000-01-05", timeframe="1h", verbose=False
    )

    assert (tmp_path / "FCHI_1d.parquet").exists()
    assert (tmp_path / "FCHI_1h.parquet").exists()
    assert (tmp_path / "FCHI_1d.parquet") != (tmp_path / "FCHI_1h.parquet")


def test_stack_multiple_tickers_single_call(tmp_path):
    """
    Multiple tickers request should return a tuple of DataFrames aligned to input order.
    """
    df_a = _mk_ohlcv("2000-01-01", "2000-01-10")
    df_b = _mk_ohlcv("2000-01-01", "2000-01-10")

    provider = DummyProvider({"^FCHI": df_a, "^STOXX": df_b})

    ds = DataStackBuilder().with_provider(provider).with_parquet_cache(tmp_path).build()

    out = ds.get_ohlcv(
        ["^FCHI", "^STOXX"],
        start="2000-01-03",
        end="2000-01-06",
        timeframe="1d",
        verbose=False,
    )

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].index.min() == pd.Timestamp("2000-01-03")
    assert out[0].index.max() == pd.Timestamp("2000-01-06")
    assert out[1].index.min() == pd.Timestamp("2000-01-03")
    assert out[1].index.max() == pd.Timestamp("2000-01-06")

    # cache files exist for both tickers
    assert (tmp_path / "FCHI_1d.parquet").exists()
    assert (tmp_path / "STOXX_1d.parquet").exists()
