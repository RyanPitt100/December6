# trend_wf_logic.py
"""
WF-style trade generation logic for Trend_EMA_Pullback strategy.

This module mirrors the trade generation behavior from walkforward_multi_instrument.py
and run_experiments.py (backtest_signals function) without modifying their outputs.

The goal is to provide a reusable function that produces the same trades as the WF pipeline
for use in live_sim_trend_portfolio.py.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd

from strategies import TrendEMAPullback, Signal


@dataclass
class WFStyleTrade:
    """
    Represents a complete trade matching the WF OOS trade CSV format.

    Attributes match columns in walkforward_{INSTRUMENT}_TrendEMAPullback_OOS_trades.csv:
        entry_time, exit_time, direction, entry, exit, sl, tp, outcome, R
    """
    instrument: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str     # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    outcome: str       # "TP", "SL", or "EXP" (expired/time stop)
    r_per_trade: float  # matches the 'R' column in WF OOS CSVs


def backtest_signals_wf_style(
    df: pd.DataFrame,
    signals: List[Signal],
    max_hold_bars: int = 96,  # 96*15m = 24h
) -> List[WFStyleTrade]:
    """
    Bar-by-bar backtest matching the exact logic from run_experiments.backtest_signals().

    This is a direct port of the WF backtest logic:
    - Enter at the first bar >= signal.time at close price
    - Walk forward up to max_hold_bars
    - LONG: if low <= SL -> SL, elif high >= TP -> TP
    - SHORT: if high >= SL -> SL, elif low <= TP -> TP
    - If neither hit, exit at last bar close ("EXP")

    Args:
        df: DataFrame with OHLC data (index = timestamp)
        signals: List of Signal objects from TrendEMAPullback.generate_signals()
        max_hold_bars: Maximum bars to hold (default 96 = 24h for 15m bars)

    Returns:
        List of WFStyleTrade objects
    """
    trades = []
    idx = df.index

    for sig in signals:
        # Find entry bar (first bar >= signal.time)
        pos = idx.searchsorted(sig.time)
        if pos >= len(idx):
            continue

        dirn = sig.direction.upper()
        sl = float(sig.stop_loss)
        tp = float(sig.take_profit)

        entry_time = idx[pos]
        entry_price = float(df.loc[entry_time, "close"])

        exit_time = None
        exit_price = None
        outcome = "EXP"

        # Walk forward checking for TP/SL hits
        for step in range(1, max_hold_bars + 1):
            j = pos + step
            if j >= len(idx):
                break
            t = idx[j]
            bar = df.loc[t]
            high = float(bar["high"])
            low = float(bar["low"])

            if dirn == "LONG":
                hit_sl = low <= sl
                hit_tp = high >= tp
                if hit_sl:
                    exit_time = t
                    exit_price = sl
                    outcome = "SL"
                    break
                if hit_tp:
                    exit_time = t
                    exit_price = tp
                    outcome = "TP"
                    break
            else:  # SHORT
                hit_sl = high >= sl
                hit_tp = low <= tp
                if hit_sl:
                    exit_time = t
                    exit_price = sl
                    outcome = "SL"
                    break
                if hit_tp:
                    exit_time = t
                    exit_price = tp
                    outcome = "TP"
                    break

        # If no TP/SL hit, exit at max hold time
        if exit_time is None:
            j = min(pos + max_hold_bars, len(idx) - 1)
            exit_time = idx[j]
            exit_price = float(df.loc[exit_time, "close"])
            outcome = "EXP"

        # Calculate R (matching WF logic exactly)
        if dirn == "LONG":
            risk_per_unit = entry_price - sl
            pnl_per_unit = exit_price - entry_price
        else:
            risk_per_unit = sl - entry_price
            pnl_per_unit = entry_price - exit_price

        R = pnl_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

        trades.append(
            WFStyleTrade(
                instrument=sig.instrument,
                entry_time=entry_time,
                exit_time=exit_time,
                direction=dirn,
                entry_price=entry_price,
                exit_price=exit_price,
                sl_price=sl,
                tp_price=tp,
                outcome=outcome,
                r_per_trade=R,
            )
        )

    return trades


def generate_trend_ema_pullback_trades_from_df(
    instrument: str,
    df: pd.DataFrame,
    max_hold_bars: int = 96,
) -> List[WFStyleTrade]:
    """
    Generate WF-style trades for Trend_EMA_Pullback strategy from a DataFrame.

    This replicates the complete WF pipeline:
    1. Generate signals using TrendEMAPullback.generate_signals()
    2. Backtest signals using the exact WF logic (TP/SL/time stop)
    3. Return list of completed trades with entry/exit/R

    Args:
        instrument: Instrument symbol (e.g., "USDJPY")
        df: Multi-timeframe DataFrame from build_multi_tf_frame()
        max_hold_bars: Maximum bars to hold (default 96 = 24h)

    Returns:
        List of WFStyleTrade objects matching WF OOS trade CSV format
    """
    # 1) Generate signals using canonical TrendEMAPullback strategy
    strategy = TrendEMAPullback()
    signals = strategy.generate_signals(df, instrument=instrument)

    # 2) Backtest signals using WF-style logic
    trades = backtest_signals_wf_style(df, signals, max_hold_bars=max_hold_bars)

    return trades
