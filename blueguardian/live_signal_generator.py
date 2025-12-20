#!/usr/bin/env python3
"""
live_signal_generator.py

Shared signal generation logic for TrendEMAPullback strategy that works consistently
across walk-forward, live-sim, and live MT5 execution.

This module provides the canonical source of truth for generating trading signals,
ensuring identical behavior in all three modes:
- Walk-forward backtesting (walkforward_multi_instrument.py)
- Historical live simulation (live_sim_trend_portfolio.py)
- Live MT5 execution (live_mt5_eval_runner.py)

The key function `generate_trend_ema_pullback_signals` takes multi-timeframe data
and produces TradeSignal objects that can be passed directly to the portfolio controller.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from strategies import TrendEMAPullback, Signal, TrendEMAPullbackParams
from portfolio_controller import TradeSignal


def generate_trend_ema_pullback_signals(
    instrument: str,
    mtf_data: pd.DataFrame,
    current_ts: pd.Timestamp,
    params: Optional[TrendEMAPullbackParams] = None,
) -> List[TradeSignal]:
    """
    Generate TrendEMAPullback signals for a single instrument at a specific timestamp.

    This function is the canonical source of truth for signal generation, used by:
    - Walk-forward backtesting (via strategies.TrendEMAPullback.generate_signals)
    - Historical live simulation (via trend_wf_logic.generate_trend_ema_pullback_trades_from_df)
    - Live MT5 execution (via this function directly)

    The function operates on fully closed bars only (no lookahead bias) and includes
    the same regime gating logic as used in the WF pipeline.

    Args:
        instrument: Symbol name (e.g., "EURUSD", "JP225.cash")
        mtf_data: Multi-timeframe DataFrame from build_multi_tf_frame() with columns:
                  - OHLC: open, high, low, close
                  - Regime labels: regime_h1, regime_h4, regime_d1
                  - Optional: range_score_h1
        current_ts: Current timestamp (use latest completed bar time for live trading)
        params: Optional TrendEMAPullbackParams (uses defaults if None)

    Returns:
        List of TradeSignal objects ready for portfolio_controller.decide_portfolio_orders()

    Implementation notes:
        - Only processes data up to and including current_ts (no lookahead)
        - Uses the exact same TrendEMAPullback strategy logic as WF/live-sim
        - Converts Signal objects (from strategies.py) to TradeSignal objects (for portfolio_controller.py)
        - Regime filtering: Requires both H1 and H4 to be "TRENDING"
        - Returns empty list if no signals at current_ts
    """
    # Slice data up to current_ts only (no lookahead)
    if current_ts not in mtf_data.index:
        # Current timestamp not in data - return empty
        return []

    # Get data up to and including current_ts
    # CRITICAL: Include sufficient history for indicators to stabilize
    # EMA50 needs ~100-150 bars, ATR needs ~20 bars, RSI needs ~14 bars
    # We use 200 bars minimum to ensure indicators match WF path values
    MIN_HISTORY_BARS = 200

    current_idx = mtf_data.index.get_loc(current_ts)
    start_idx = max(0, current_idx - MIN_HISTORY_BARS)
    data_slice = mtf_data.iloc[start_idx:current_idx + 1].copy()

    # Ensure we have enough history for indicators
    if len(data_slice) < 100:
        # Not enough data for EMA50 + ATR + RSI
        return []

    # Generate signals using canonical TrendEMAPullback strategy
    strategy = TrendEMAPullback(params=params)
    signals: List[Signal] = strategy.generate_signals(data_slice, instrument=instrument)

    # Filter signals to only those at current_ts
    # (The strategy generates signals across all bars in data_slice, but we only want
    # signals that trigger at current_ts for bar-by-bar live execution)
    current_ts_signals = [sig for sig in signals if sig.time == current_ts]

    # Convert Signal objects to TradeSignal objects for portfolio controller
    trade_signals: List[TradeSignal] = []

    for sig in current_ts_signals:
        trade_signal = TradeSignal(
            instrument=sig.instrument,
            direction=sig.direction.lower(),  # Convert "LONG"/"SHORT" to "long"/"short"
            entry_price=sig.entry_price,
            sl_price=sig.stop_loss,
            tp_price=sig.take_profit,
            reason=f"trend_ema_pullback_{sig.direction.lower()}_entry"
        )
        trade_signals.append(trade_signal)

    return trade_signals


def generate_signals_for_portfolio(
    instrument_data: Dict[str, pd.DataFrame],
    current_ts: pd.Timestamp,
    portfolio_instruments: List[str],
    params: Optional[TrendEMAPullbackParams] = None,
) -> List[TradeSignal]:
    """
    Generate signals for all instruments in a portfolio at a specific timestamp.

    This is a convenience wrapper around generate_trend_ema_pullback_signals() that
    processes multiple instruments and aggregates their signals.

    Args:
        instrument_data: Dict mapping symbol -> multi-timeframe DataFrame
        current_ts: Current timestamp (use latest completed bar time)
        portfolio_instruments: List of symbols to generate signals for
        params: Optional strategy parameters

    Returns:
        List of TradeSignal objects for all instruments

    Usage:
        # In live MT5 runner
        signals = generate_signals_for_portfolio(
            instrument_data=instrument_data,
            current_ts=latest_bar_time,
            portfolio_instruments=['EURUSD', 'USDJPY', 'JP225.cash', ...],
        )
        # Pass signals to portfolio controller
        orders = decide_portfolio_orders(account_state, signals, ...)
    """
    all_signals: List[TradeSignal] = []

    for symbol in portfolio_instruments:
        if symbol not in instrument_data:
            continue

        df = instrument_data[symbol]

        # Generate signals for this instrument
        signals = generate_trend_ema_pullback_signals(
            instrument=symbol,
            mtf_data=df,
            current_ts=current_ts,
            params=params,
        )

        all_signals.extend(signals)

    return all_signals
