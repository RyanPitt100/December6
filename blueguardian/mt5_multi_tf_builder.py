#!/usr/bin/env python3
"""
mt5_multi_tf_builder.py

Build multi-timeframe data with regime labels from live MT5 data.

This module bridges MT5 live data to the multi-TF structure expected by the strategy,
including real-time regime labeling for H1, H4, and D1 timeframes.

IMPORTANT: Uses pre-computed regime thresholds from regime_thresholds.yml to ensure
consistency with historical backtests. The thresholds were extracted from full
historical data using 40th/60th percentile quantiles.
"""

from typing import Optional, Dict, Any
import os
import pandas as pd
import yaml

import mt5_bridge
from regime_labeler import label_regimes, RegimeParams


# Cache for regime thresholds (loaded once)
_regime_thresholds_cache: Optional[Dict[str, Any]] = None


def _load_regime_thresholds() -> Dict[str, Any]:
    """Load pre-computed regime thresholds from YAML config."""
    global _regime_thresholds_cache

    if _regime_thresholds_cache is not None:
        return _regime_thresholds_cache

    # Look for config in current directory, parent directory, or script directory
    config_paths = [
        "regime_thresholds.yml",
        "../regime_thresholds.yml",
        os.path.join(os.path.dirname(__file__), "..", "regime_thresholds.yml"),
    ]

    for path in config_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                _regime_thresholds_cache = yaml.safe_load(f)
            return _regime_thresholds_cache

    # Return empty dict if not found (will use defaults)
    print("[WARNING] regime_thresholds.yml not found, using default quantile-based thresholds")
    _regime_thresholds_cache = {"thresholds": {}}
    return _regime_thresholds_cache


def _get_regime_params_for_instrument(symbol: str, timeframe: str) -> RegimeParams:
    """
    Get RegimeParams with pre-computed thresholds for a specific instrument/timeframe.

    Falls back to default params if instrument not found in config.
    """
    config = _load_regime_thresholds()
    thresholds = config.get("thresholds", {})

    # Try exact match first, then without .cash suffix
    inst_config = thresholds.get(symbol) or thresholds.get(symbol.replace(".cash", ""))

    if inst_config is None or timeframe not in inst_config:
        # Fallback: use fixed thresholds (not quantile-based) for consistency
        return RegimeParams(
            use_quantiles=False,
            er_range_max=0.25,
            er_trend_min=0.35,
            adx_range_max=20.0,
            adx_trend_min=25.0,
        )

    tf_config = inst_config[timeframe]

    # Create params with pre-computed thresholds (disable quantile calculation)
    return RegimeParams(
        use_quantiles=False,
        er_range_max=tf_config["er_range_max"],
        er_trend_min=tf_config["er_trend_min"],
        adx_range_max=tf_config["adx_range_max"],
        adx_trend_min=tf_config["adx_trend_min"],
    )


def build_multi_tf_from_mt5(
    symbol: str,
    n_bars_15m: int = 500,
) -> Optional[pd.DataFrame]:
    """
    Build multi-timeframe DataFrame from live MT5 data with regime labels.

    This function fetches OHLC data for multiple timeframes from MT5 and merges them
    into a single 15m-based DataFrame with regime labels for H1, H4, and D1.

    The output structure matches multi_tf_builder.build_multi_tf_frame() so that
    the same strategy code works for both offline (historical CSV) and live (MT5) data.

    Args:
        symbol: MT5 symbol name (e.g., "EURUSD", "JP225.cash")
        n_bars_15m: Number of 15m bars to fetch (default: 500)

    Returns:
        DataFrame with 15m timestamp index and columns:
        - open, high, low, close (15m OHLC)
        - regime_h1, range_score_h1
        - regime_h4
        - regime_d1

        Returns None if data fetch fails.

    Usage:
        df = build_multi_tf_from_mt5("EURUSD", n_bars_15m=500)
        if df is not None:
            signals = generate_trend_ema_pullback_signals("EURUSD", df, current_ts)
    """
    # Fetch 15m base data
    df_15m = mt5_bridge.fetch_ohlc(symbol, "15m", n_bars=n_bars_15m)
    if df_15m is None:
        return None

    # Fetch higher timeframes for regime labeling
    # We need enough bars to compute regime indicators
    # H1: 500 15m bars = ~125 H1 bars (need ~200 for indicators)
    df_h1 = mt5_bridge.fetch_ohlc(symbol, "1h", n_bars=300)
    df_h4 = mt5_bridge.fetch_ohlc(symbol, "4h", n_bars=300)
    df_d1 = mt5_bridge.fetch_ohlc(symbol, "1d", n_bars=300)

    if df_h1 is None or df_h4 is None or df_d1 is None:
        # Fallback: return 15m data without regime labels
        # (Strategy will skip trading due to missing regime columns)
        return df_15m

    # Label regimes for each timeframe using pre-computed thresholds
    # This ensures consistency with historical backtests
    params_h1 = _get_regime_params_for_instrument(symbol, "1h")
    params_h4 = _get_regime_params_for_instrument(symbol, "4h")
    params_d1 = _get_regime_params_for_instrument(symbol, "1d")

    df_h1_labeled = label_regimes(df_h1, params_h1)
    df_h4_labeled = label_regimes(df_h4, params_h4)
    df_d1_labeled = label_regimes(df_d1, params_d1)

    # Reset index to merge
    df_15m = df_15m.reset_index()
    df_h1_labeled = df_h1_labeled.reset_index()
    df_h4_labeled = df_h4_labeled.reset_index()
    df_d1_labeled = df_d1_labeled.reset_index()

    # Merge H1 regime labels onto 15m data (forward fill)
    df_15m = pd.merge_asof(
        df_15m.sort_values("timestamp"),
        df_h1_labeled[["timestamp", "regime", "range_score"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )
    df_15m.rename(columns={"regime": "regime_h1", "range_score": "range_score_h1"}, inplace=True)

    # Merge H4 regime labels
    df_15m = pd.merge_asof(
        df_15m.sort_values("timestamp"),
        df_h4_labeled[["timestamp", "regime"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )
    df_15m.rename(columns={"regime": "regime_h4"}, inplace=True)

    # Merge D1 regime labels
    df_15m = pd.merge_asof(
        df_15m.sort_values("timestamp"),
        df_d1_labeled[["timestamp", "regime"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )
    df_15m.rename(columns={"regime": "regime_d1"}, inplace=True)

    # Set timestamp as index
    df_15m = df_15m.set_index("timestamp").sort_index()

    # Return final DataFrame matching build_multi_tf_frame() structure
    return df_15m


def get_latest_completed_bar_time(df: pd.DataFrame, current_time: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Get the timestamp of the latest completed 15m bar.

    For live trading, we must only trade on fully formed bars to avoid lookahead bias.
    This function returns the timestamp of the most recent bar that is fully closed.

    Args:
        df: Multi-timeframe DataFrame with timestamp index
        current_time: Current wall-clock time (pd.Timestamp with timezone)

    Returns:
        Timestamp of latest completed bar, or None if no completed bars available

    Logic:
        - 15m bars close at :00, :15, :30, :45
        - Current time 10:17 -> latest completed bar is 10:15
        - Current time 10:30 -> bar at 10:30 may still be forming -> use 10:15
        - Add small buffer (30 seconds) to ensure bar is fully closed

    Usage:
        current_time = pd.Timestamp.now(tz='UTC')
        latest_bar_time = get_latest_completed_bar_time(df, current_time)
        if latest_bar_time:
            signals = generate_trend_ema_pullback_signals(symbol, df, latest_bar_time)
    """
    # Ensure current_time has timezone
    if current_time.tz is None:
        current_time = current_time.tz_localize('UTC')

    # Round down to latest 15m bar boundary
    # Subtract 30 seconds buffer to ensure bar is fully closed
    bar_boundary = current_time - pd.Timedelta(seconds=30)
    bar_boundary = bar_boundary.floor('15min')

    # Find latest bar in data that is <= bar_boundary
    available_bars = df.index[df.index <= bar_boundary]

    if len(available_bars) == 0:
        return None

    return available_bars[-1]
