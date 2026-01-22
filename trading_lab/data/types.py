from __future__ import annotations

# Yahoo intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
# We'll standardize to a canonical set.
SUPPORTED_TIMEFRAMES = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}


def validate_timeframe(tf: str) -> str:
    tf = tf.strip()
    if tf not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe '{tf}'. Supported: {sorted(SUPPORTED_TIMEFRAMES)}"
        )
    # Normalize "60m" to "1h" if you want a single canonical hourly key
    if tf == "60m":
        return "1h"
    return tf
