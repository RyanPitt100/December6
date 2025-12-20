#!/usr/bin/env python3
"""
live_sim_trend_portfolio.py

Historical live simulation runner for the Trend_EMA_Pullback portfolio with rolling FTMO evaluation cycles.

This script simulates a "live" trading environment using historical OHLC data,
applying the full risk management stack:
  - FTMO overlays (daily/total loss limits, eval profit target)
  - Risk Envelope Guard
  - Portfolio controller order generation
  - Position sizing via ATR-based risk

Key features:
  - Uses precomputed WF-style trades (aligned with walkforward pipeline)
  - Supports rolling FTMO evaluation cycles (default)
  - Each cycle starts with 200k equity and runs until PASS or FAIL
  - Tracks cycle statistics (pass rate, duration, equity gain)

Usage:
    # Rolling eval mode (default)
    python live_sim_trend_portfolio.py \
        --mode eval \
        --portfolio-id phase2_core \
        --output-equity-csv live_sim_equity.csv \
        --output-cycles-csv live_sim_cycles.csv \
        --output-summary-json live_sim_summary.json

    # Single eval mode (stop after first terminal condition)
    python live_sim_trend_portfolio.py \
        --mode eval \
        --single-eval \
        --output-equity-csv live_sim_equity.csv

Arguments:
    --mode: "eval" (default) or "funded" - determines risk per trade and eval rules
    --portfolio-id: Portfolio identifier (default: "phase2_core")
    --start-date: Optional ISO date (YYYY-MM-DD) to start simulation
    --end-date: Optional ISO date (YYYY-MM-DD) to end simulation
    --output-equity-csv: Path to save equity curve CSV (default: live_sim_equity.csv)
    --output-summary-json: Optional path to save summary statistics JSON
    --output-cycles-csv: Optional path to save cycles CSV (for rolling mode)
    --single-eval: Stop after first terminal condition (old behavior)
    --debug-risk: Enable verbose risk state logging

Output:
    - Equity curve CSV with cycle_id, timestamp, equity, balance, risk metrics
    - Cycles CSV with cycle statistics (result, duration, trades, equity gain)
    - Summary JSON with cycle aggregates (pass rate, avg duration, etc.)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import json

import pandas as pd
import numpy as np

# Import existing modules
from multi_tf_builder import build_multi_tf_frame
from strategies import TrendEMAPullback
from config_loader import load_all_configs
from risk_manager import (
    AccountState,
    Position,
    evaluate_daily_risk_state,
    describe_risk_state,
)
from portfolio_controller import (
    TradeSignal,
    Order,
    decide_portfolio_orders,
    describe_orders,
)
from trend_wf_logic import generate_trend_ema_pullback_trades_from_df, WFStyleTrade


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute ATR (Average True Range) for a DataFrame with OHLC columns.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR lookback period (default 20)

    Returns:
        Series with ATR values
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        # Fallback: use close volatility if OHLC not available
        return df['close'].rolling(period).std() * np.sqrt(period)

    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr


def load_instrument_data(
    portfolio_cfg,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load multi-timeframe data for all portfolio instruments.

    Args:
        portfolio_cfg: InstrumentPortfolioConfig with list of instruments
        start_date: Optional ISO date string to filter data
        end_date: Optional ISO date string to filter data

    Returns:
        Dict mapping instrument symbol to DataFrame with OHLC + features
    """
    instrument_data = {}

    for inst_cfg in portfolio_cfg.instruments:
        symbol = inst_cfg.symbol
        print(f"[LOAD] Loading data for {symbol}...")

        try:
            df = build_multi_tf_frame(symbol)

            # Ensure sorted by index
            df = df.sort_index()

            # Filter by date range if provided
            if start_date:
                start_dt = pd.to_datetime(start_date)
                # Handle timezone-aware index
                if df.index.tz is not None:
                    start_dt = start_dt.tz_localize(df.index.tz)
                df = df[df.index >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                # Handle timezone-aware index
                if df.index.tz is not None:
                    end_dt = end_dt.tz_localize(df.index.tz)
                df = df[df.index <= end_dt]

            # Drop rows with missing essential columns
            essential_cols = ['close']
            df = df.dropna(subset=essential_cols)

            if df.empty:
                print(f"[SKIP] {symbol}: No data after filtering")
                continue

            # Compute ATR if not already present
            if 'atr' not in df.columns:
                df['atr'] = compute_atr(df, period=20)

            instrument_data[symbol] = df
            print(f"[OK] {symbol}: {len(df)} bars loaded")

        except Exception as e:
            print(f"[ERROR] Failed to load {symbol}: {e}")
            continue

    return instrument_data


def precompute_wf_style_trades(
    instrument_data: Dict[str, pd.DataFrame]
) -> Dict[str, List[WFStyleTrade]]:
    """
    Precompute WF-style trades for all instruments using canonical WF logic.

    This replaces the old signal generation approach with a complete trade precomputation
    that matches the walkforward pipeline exactly.

    Args:
        instrument_data: Dict of symbol -> DataFrame

    Returns:
        Dict of symbol -> List[WFStyleTrade] (complete trades with entry/exit/R)
    """
    all_trades = {}

    for symbol, df in instrument_data.items():
        print(f"[WF-TRADES] Precomputing trades for {symbol}...")

        try:
            # Generate WF-style trades using canonical logic
            trades = generate_trend_ema_pullback_trades_from_df(
                instrument=symbol,
                df=df,
                max_hold_bars=96  # 24h for 15m bars
            )

            all_trades[symbol] = trades

            # Count trade types for sanity check
            n_long = sum(1 for t in trades if t.direction == "LONG")
            n_short = sum(1 for t in trades if t.direction == "SHORT")
            n_tp = sum(1 for t in trades if t.outcome == "TP")
            n_sl = sum(1 for t in trades if t.outcome == "SL")
            n_exp = sum(1 for t in trades if t.outcome == "EXP")

            print(f"  [OK] {symbol}: {len(trades)} trades ({n_long} long, {n_short} short)")
            print(f"       Outcomes: {n_tp} TP, {n_sl} SL, {n_exp} EXP")

        except Exception as e:
            print(f"[ERROR] Failed to generate trades for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            all_trades[symbol] = []

    return all_trades


def build_time_grid(
    instrument_data: Dict[str, pd.DataFrame]
) -> List[pd.Timestamp]:
    """
    Build unified sorted time grid across all instruments.

    Args:
        instrument_data: Dict of symbol -> DataFrame

    Returns:
        Sorted list of unique timestamps
    """
    all_times = set()
    for df in instrument_data.values():
        all_times.update(df.index)

    return sorted(all_times)


def compute_position_pnl(
    position: Position,
    current_price: float,
    initial_equity: float,
) -> float:
    """
    Return current PnL for a single position in account currency (USD).

    Args:
        position: Position object with entry_price, sl_price, direction, risk_pct
        current_price: Current market price for the instrument
        initial_equity: Initial equity to calculate capital at risk

    Returns:
        PnL in currency (float)
    """
    # Compute price move based on direction
    if position.direction == "long":
        price_move = current_price - position.entry_price
    else:  # short
        price_move = position.entry_price - current_price

    # Compute 1R move
    r_price_move = abs(position.entry_price - position.sl_price)
    if r_price_move <= 0:
        return 0.0

    # Compute R multiple
    r_multiple = price_move / r_price_move

    # Capital at risk = initial_equity * risk_pct / 100
    risk_fraction = position.risk_pct / 100.0
    capital_at_risk = initial_equity * risk_fraction

    # PnL in currency
    pnl = r_multiple * capital_at_risk

    return pnl


def mark_positions_to_market(
    open_positions: List[Position],
    instrument_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    initial_equity: float,
) -> float:
    """
    Compute unrealized PnL for all open positions at current time.

    Args:
        open_positions: List of open Position objects
        instrument_data: Dict of symbol -> DataFrame
        current_time: Current timestamp
        initial_equity: Initial equity to compute absolute PnL

    Returns:
        Total unrealized PnL in currency (absolute value)
    """
    total_unrealized_pnl = 0.0

    for pos in open_positions:
        if pos.instrument not in instrument_data:
            continue

        df = instrument_data[pos.instrument]

        # Get current price (use close, or fallback to last known)
        if current_time in df.index:
            current_price = df.loc[current_time, 'close']
        else:
            # Use last available price before current_time
            prior_data = df[df.index <= current_time]
            if prior_data.empty:
                continue
            current_price = prior_data.iloc[-1]['close']

        # Compute PnL using helper
        pnl = compute_position_pnl(pos, current_price, initial_equity)
        total_unrealized_pnl += pnl

    return total_unrealized_pnl


def create_trade_signals_at_bar_from_wf_trades(
    open_positions: List[Position],
    wf_trades: Dict[str, List[WFStyleTrade]],
    current_time: pd.Timestamp,
) -> Tuple[List[TradeSignal], List[Order]]:
    """
    Generate TradeSignal objects for entries and Order objects for exits using precomputed WF trades.

    This replaces the old on-the-fly signal generation with a replay of precomputed WF trades.
    The WF trades act as the "script" for when to enter/exit, and the risk stack sits on top.

    Args:
        open_positions: Current open positions
        wf_trades: Dict of symbol -> List[WFStyleTrade] (precomputed)
        current_time: Current timestamp

    Returns:
        Tuple of (entry_signals, exit_orders)
    """
    entry_signals = []
    exit_orders = []

    # Track which instruments have open positions and which WF trade they correspond to
    open_instruments = {pos.instrument: pos for pos in open_positions}

    # Check each instrument for WF trades that should trigger now
    for symbol, trades in wf_trades.items():
        current_position = open_instruments.get(symbol)

        for trade in trades:
            # Check for entry signal (WF trade entry_time matches current bar)
            if trade.entry_time == current_time and current_position is None:
                # Create entry signal matching the WF trade
                entry_signals.append(TradeSignal(
                    instrument=symbol,
                    direction=trade.direction.lower(),  # Convert to lowercase for consistency
                    entry_price=trade.entry_price,
                    sl_price=trade.sl_price,
                    tp_price=trade.tp_price,
                    reason=f"wf_replay_{trade.direction.lower()}_entry"
                ))

            # Check for exit signal (WF trade exit_time matches current bar)
            elif trade.exit_time == current_time and current_position is not None:
                # Only exit if we're actually in this trade
                # (Need to check that current position matches this WF trade)
                # For simplicity, we assume one position per instrument at a time
                exit_orders.append(Order(
                    instrument=symbol,
                    action="close",
                    direction=None,
                    size_lots=current_position.size_lots,
                    entry_price=trade.exit_price,  # Use WF exit price
                    sl_price=None,
                    tp_price=None,
                    reason=f"wf_replay_{trade.outcome.lower()}_exit"
                ))

    return entry_signals, exit_orders


def apply_orders(
    orders: List[Order],
    open_positions: List[Position],
    balance: float,
    realised_pnl_today: float,
    initial_equity: float,
    trade_log: Optional[List[Dict[str, Any]]] = None,
    current_time: Optional[pd.Timestamp] = None,
) -> Tuple[List[Position], float, float, int]:
    """
    Apply orders to update open positions and balance.

    Args:
        orders: List of Order objects from portfolio controller
        open_positions: Current list of open positions
        balance: Current realised balance
        realised_pnl_today: Realised PnL for current day
        initial_equity: Initial equity for PnL calculation
        trade_log: Optional list to append trade records to
        current_time: Optional current timestamp for logging

    Returns:
        Tuple of (updated_positions, updated_balance, updated_pnl_today, trade_count)
    """
    trade_count = 0

    for order in orders:
        if order.action == "open":
            # Open new position
            new_position = Position(
                instrument=order.instrument,
                direction=order.direction,
                entry_price=order.entry_price,
                sl_price=order.sl_price,
                size_lots=order.size_lots,
                risk_pct=0.5,  # Will be set properly by portfolio controller sizing
            )
            open_positions.append(new_position)
            trade_count += 1

            # Log entry
            if trade_log is not None and current_time is not None:
                trade_log.append({
                    "timestamp": current_time,
                    "instrument": order.instrument,
                    "action": "OPEN",
                    "direction": order.direction,
                    "entry_price": order.entry_price,
                    "sl_price": order.sl_price,
                    "exit_price": None,
                    "r_multiple": None,
                    "pnl": None,
                    "risk_pct": 0.5,
                    "reason": order.reason,
                })

        elif order.action == "close":
            # Close matching position(s)
            positions_to_remove = []

            for i, pos in enumerate(open_positions):
                if pos.instrument == order.instrument:
                    # Compute realised PnL
                    if order.entry_price is not None:
                        # Use provided exit price
                        exit_price = order.entry_price
                    else:
                        # Should not happen, but fallback
                        continue

                    # Compute price move
                    if pos.direction == "long":
                        price_move = exit_price - pos.entry_price
                    else:
                        price_move = pos.entry_price - exit_price

                    # Compute R multiple
                    r_price_move = abs(pos.entry_price - pos.sl_price)
                    r_multiple = 0.0
                    pnl_absolute = 0.0

                    if r_price_move > 0:
                        r_multiple = price_move / r_price_move
                        pnl_pct = r_multiple * pos.risk_pct
                        pnl_absolute = (pnl_pct / 100.0) * initial_equity

                        balance += pnl_absolute
                        realised_pnl_today += pnl_absolute

                    positions_to_remove.append(i)
                    trade_count += 1

                    # Log exit
                    if trade_log is not None and current_time is not None:
                        trade_log.append({
                            "timestamp": current_time,
                            "instrument": order.instrument,
                            "action": "CLOSE",
                            "direction": pos.direction,
                            "entry_price": pos.entry_price,
                            "sl_price": pos.sl_price,
                            "exit_price": exit_price,
                            "r_multiple": r_multiple,
                            "pnl": pnl_absolute,
                            "risk_pct": pos.risk_pct,
                            "reason": order.reason,
                        })

            # Remove closed positions
            for i in sorted(positions_to_remove, reverse=True):
                open_positions.pop(i)

    return open_positions, balance, realised_pnl_today, trade_count


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_single_eval_cycle(
    cycle_id: int,
    mode: str,
    time_grid: List[pd.Timestamp],
    start_idx: int,
    initial_equity: float,
    wf_trades: Dict[str, List[WFStyleTrade]],
    instrument_data: Dict[str, pd.DataFrame],
    portfolio_cfg: Any,
    risk_cfg: Any,
    ftmo_cfg: Any,
    debug_risk: bool,
) -> Tuple[List[Dict], List[Dict], int, str, Optional[pd.Timestamp]]:
    """
    Run a single FTMO evaluation cycle starting from start_idx in time_grid.

    This function runs the simulation from start_idx until a terminal condition is hit
    (PASS, FAIL_DAILY, FAIL_TOTAL, FAIL_ENVELOPE, FAIL_HARD_BRAKE) or the time grid ends.

    Args:
        cycle_id: Cycle identifier (1, 2, 3, ...)
        mode: "eval" or "funded"
        time_grid: Full list of timestamps
        start_idx: Index in time_grid to start this cycle
        initial_equity: Starting equity for this cycle
        wf_trades: Precomputed WF-style trades
        instrument_data: Instrument DataFrames
        portfolio_cfg: Portfolio configuration
        risk_cfg: Risk configuration
        ftmo_cfg: FTMO overlay configuration
        debug_risk: Enable verbose risk logging

    Returns:
        Tuple of:
        - equity_rows: List of equity curve dicts for this cycle
        - trade_log: List of trade dicts for this cycle
        - end_idx: Index in time_grid where cycle ended
        - terminal_reason: Terminal condition string ("PASS", "FAIL_DAILY", etc., or "INCOMPLETE")
        - end_time: Timestamp where cycle ended (or None if incomplete)
    """
    # Initialize cycle state
    balance = initial_equity
    equity = initial_equity
    start_of_day_equity = initial_equity
    realised_pnl_today = 0.0
    open_positions: List[Position] = []

    current_date = None

    equity_rows = []
    trade_log = []

    terminal_condition_hit = False
    terminal_reason = ""
    actual_end_time = None
    end_idx = start_idx

    # Cycle simulation loop
    for i in range(start_idx, len(time_grid)):
        current_time = time_grid[i]
        end_idx = i

        # Check for day boundary
        bar_date = current_time.date()
        if current_date is None or bar_date != current_date:
            # New day
            current_date = bar_date
            # Update start_of_day_equity with current balance + open PnL
            start_of_day_equity = balance + mark_positions_to_market(
                open_positions, instrument_data, current_time, initial_equity
            )
            realised_pnl_today = 0.0

        # Mark positions to market - equity is ALWAYS derived, never incremented
        open_pnl = mark_positions_to_market(
            open_positions, instrument_data, current_time, initial_equity
        )
        equity = balance + open_pnl

        # Build AccountState
        account_state = AccountState(
            equity=equity,
            balance=balance,
            start_of_day_equity=start_of_day_equity,
            realised_pnl_today=realised_pnl_today,
            open_positions=open_positions,
        )

        # Evaluate risk state
        risk_state = evaluate_daily_risk_state(account_state, ftmo_cfg, mode=mode)

        # Debug logging
        if debug_risk and (
            risk_state["combined_risk_pct"] > 2.0
            or risk_state.get("hit_eval_profit_target", False)
            or risk_state.get("envelope_guard_triggered", False)
        ):
            print(f"\n[DEBUG CYCLE {cycle_id}] {current_time}")
            print(describe_risk_state(risk_state))

        # Check for hard stop conditions
        hard_stop = (
            risk_state.get("breached_total_limit", False)
            or risk_state.get("breached_daily_limit", False)
            or risk_state.get("hard_brake_triggered", False)
            or risk_state.get("envelope_guard_triggered", False)
            or risk_state.get("hit_eval_profit_target", False)
        )

        if hard_stop:
            # Flatten all positions at current prices
            for pos in open_positions:
                df = instrument_data.get(pos.instrument)
                if df is None:
                    continue

                # Get current price
                if current_time in df.index:
                    current_price = df.loc[current_time, 'close']
                else:
                    prior_data = df[df.index <= current_time]
                    if prior_data.empty:
                        continue
                    current_price = prior_data.iloc[-1]['close']

                # Realize PnL
                pnl = compute_position_pnl(pos, current_price, initial_equity)
                balance += pnl
                realised_pnl_today += pnl

            # Clear positions
            open_positions.clear()

            # Recalculate equity (should equal balance now)
            equity = balance + 0.0  # No open PnL

            # Determine terminal reason
            if risk_state.get("hit_eval_profit_target", False):
                terminal_reason = "PASS"
                if debug_risk:
                    print(f"\n[CYCLE {cycle_id} TERMINAL] {current_time}: Eval profit target hit at ${balance:,.2f}")
            elif risk_state.get("breached_total_limit", False):
                terminal_reason = "FAIL_TOTAL"
                if debug_risk:
                    print(f"\n[CYCLE {cycle_id} TERMINAL] {current_time}: Total loss limit breached")
            elif risk_state.get("breached_daily_limit", False):
                terminal_reason = "FAIL_DAILY"
                if debug_risk:
                    print(f"\n[CYCLE {cycle_id} TERMINAL] {current_time}: Daily loss limit breached")
            elif risk_state.get("envelope_guard_triggered", False):
                terminal_reason = "FAIL_ENVELOPE"
                if debug_risk:
                    print(f"\n[CYCLE {cycle_id} TERMINAL] {current_time}: Risk envelope guard triggered")
            elif risk_state.get("hard_brake_triggered", False):
                terminal_reason = "FAIL_HARD_BRAKE"
                if debug_risk:
                    print(f"\n[CYCLE {cycle_id} TERMINAL] {current_time}: Hard brake triggered")
            else:
                terminal_reason = "FAIL_OTHER"
                if debug_risk:
                    print(f"\n[CYCLE {cycle_id} TERMINAL] {current_time}: Other hard stop condition")

            if debug_risk:
                print(describe_risk_state(risk_state))
                print(f"[CYCLE {cycle_id} TERMINAL] Positions flattened. Final balance: ${balance:,.2f}, Final equity: ${equity:,.2f}")

            # Sanity checks
            assert len(open_positions) == 0, f"Expected 0 positions after flattening, got {len(open_positions)}"
            assert abs(equity - balance) < 1e-6, f"Equity ({equity}) and balance ({balance}) should be equal after flattening"

            # Log final equity row
            equity_rows.append({
                "cycle_id": cycle_id,
                "timestamp": current_time,
                "equity": equity,
                "balance": balance,
                "realised_pnl_today": realised_pnl_today,
                "n_open_positions": len(open_positions),
                "dd_today_pct": risk_state["dd_today_pct"],
                "open_risk_pct": risk_state["open_risk_pct"],
                "combined_risk_pct": risk_state["combined_risk_pct"],
                "total_dd_pct": risk_state["total_dd_pct"],
                "hit_eval_profit_target": risk_state.get("hit_eval_profit_target", False),
                "envelope_guard_triggered": risk_state.get("envelope_guard_triggered", False),
                "terminal_condition": terminal_reason,
            })

            terminal_condition_hit = True
            actual_end_time = current_time
            break  # End this cycle

        # Normal trading - generate entry signals and exit orders from precomputed WF trades
        entry_signals, exit_orders = create_trade_signals_at_bar_from_wf_trades(
            open_positions, wf_trades, current_time
        )

        # Process exits first (always execute)
        if exit_orders:
            open_positions, balance, realised_pnl_today, exits_executed = apply_orders(
                exit_orders, open_positions, balance, realised_pnl_today, initial_equity,
                trade_log=trade_log, current_time=current_time
            )

        # Get entry orders from portfolio controller (respects risk limits)
        entry_orders = decide_portfolio_orders(
            account_state=account_state,
            signals=entry_signals,
            portfolio_cfg=portfolio_cfg,
            risk_cfg=risk_cfg,
            ftmo_cfg=ftmo_cfg,
            mode=mode,
        )

        # Apply entry orders (modifies balance and open_positions only)
        if entry_orders:
            open_positions, balance, realised_pnl_today, entries_executed = apply_orders(
                entry_orders, open_positions, balance, realised_pnl_today, initial_equity,
                trade_log=trade_log, current_time=current_time
            )

        # Recalculate equity after orders (always derived)
        open_pnl = mark_positions_to_market(
            open_positions, instrument_data, current_time, initial_equity
        )
        equity = balance + open_pnl

        # Log equity curve
        equity_rows.append({
            "cycle_id": cycle_id,
            "timestamp": current_time,
            "equity": equity,
            "balance": balance,
            "realised_pnl_today": realised_pnl_today,
            "n_open_positions": len(open_positions),
            "dd_today_pct": risk_state["dd_today_pct"],
            "open_risk_pct": risk_state["open_risk_pct"],
            "combined_risk_pct": risk_state["combined_risk_pct"],
            "total_dd_pct": risk_state["total_dd_pct"],
            "hit_eval_profit_target": risk_state.get("hit_eval_profit_target", False),
            "envelope_guard_triggered": risk_state.get("envelope_guard_triggered", False),
            "terminal_condition": terminal_reason if terminal_condition_hit else "",
        })

    # If we finished the grid without hitting a terminal condition
    if not terminal_condition_hit:
        terminal_reason = "INCOMPLETE"
        actual_end_time = time_grid[-1]

    return equity_rows, trade_log, end_idx, terminal_reason, actual_end_time


def run_simulation(
    mode: str,
    portfolio_id: str,
    start_date: Optional[str],
    end_date: Optional[str],
    output_equity_csv: str,
    output_summary_json: Optional[str],
    output_cycles_csv: Optional[str],
    single_eval: bool,
    debug_risk: bool,
) -> None:
    """
    Run historical live simulation with optional rolling FTMO evaluation cycles.

    Args:
        mode: "eval" or "funded"
        portfolio_id: Portfolio identifier
        start_date: Optional start date (ISO string)
        end_date: Optional end date (ISO string)
        output_equity_csv: Path to save equity curve
        output_summary_json: Optional path to save summary stats
        output_cycles_csv: Optional path to save cycles CSV (for rolling mode)
        single_eval: If True, stop after first terminal condition (old behavior)
        debug_risk: Enable verbose risk logging
    """
    print("=" * 80)
    print("LIVE SIMULATION - TREND_EMA_PULLBACK PORTFOLIO")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Portfolio: {portfolio_id}")
    print(f"Date range: {start_date or 'all'} to {end_date or 'all'}")
    print(f"Rolling eval: {'No (single cycle)' if single_eval else 'Yes (rolling cycles)'}")
    print("=" * 80)

    # 1) Load configs
    print("\n[1/7] Loading configurations...")
    portfolio_cfg, risk_cfg, ftmo_cfg = load_all_configs()

    if portfolio_cfg.portfolio_id != portfolio_id:
        print(f"[ERROR] Portfolio ID mismatch: config has '{portfolio_cfg.portfolio_id}', requested '{portfolio_id}'")
        sys.exit(1)

    print(f"[OK] Loaded portfolio: {portfolio_cfg.portfolio_id}")
    print(f"[OK] Instruments: {[inst.symbol for inst in portfolio_cfg.instruments]}")
    print(f"[OK] Risk per trade ({mode}): {risk_cfg.risk_per_trade_eval_pct if mode == 'eval' else risk_cfg.risk_per_trade_funded_pct}%")

    # 2) Load instrument data
    print("\n[2/7] Loading instrument data...")
    instrument_data = load_instrument_data(portfolio_cfg, start_date, end_date)

    if not instrument_data:
        print("[ERROR] No instrument data loaded. Exiting.")
        sys.exit(1)

    print(f"[OK] Loaded {len(instrument_data)} instruments")

    # 3) Precompute WF-style trades
    print("\n[3/7] Precomputing WF-style trades...")
    wf_trades = precompute_wf_style_trades(instrument_data)

    # 4) Build time grid
    print("\n[4/7] Building unified time grid...")
    time_grid = build_time_grid(instrument_data)
    print(f"[OK] Time grid: {len(time_grid)} timestamps from {time_grid[0]} to {time_grid[-1]}")

    # 5) Determine initial equity
    print("\n[5/7] Determining initial equity...")

    if mode == "eval":
        initial_equity = ftmo_cfg.eval_phase.starting_equity if ftmo_cfg.eval_phase else 200000.0
    else:
        initial_equity = 200000.0  # Default for funded

    print(f"[OK] Initial equity: ${initial_equity:,.2f}")
    print(f"[OK] Mode: {mode}")

    # 6) Run simulation with rolling cycles or single cycle
    print("\n[6/7] Running simulation...")
    print("-" * 80)

    all_equity_rows = []
    all_trade_logs = []
    all_cycles = []

    cycle_id = 1
    next_start_idx = 0

    while next_start_idx < len(time_grid):
        start_time = time_grid[next_start_idx]
        print(f"\n[CYCLE {cycle_id}] Starting at {start_time} (idx {next_start_idx}/{len(time_grid)})")

        # Run single cycle
        equity_rows, trade_log, end_idx, terminal_reason, end_time = run_single_eval_cycle(
            cycle_id=cycle_id,
            mode=mode,
            time_grid=time_grid,
            start_idx=next_start_idx,
            initial_equity=initial_equity,
            wf_trades=wf_trades,
            instrument_data=instrument_data,
            portfolio_cfg=portfolio_cfg,
            risk_cfg=risk_cfg,
            ftmo_cfg=ftmo_cfg,
            debug_risk=debug_risk,
        )

        # Collect equity rows and trades
        all_equity_rows.extend(equity_rows)
        all_trade_logs.extend(trade_log)

        # Compute cycle statistics
        n_trades = len(trade_log)
        if equity_rows:
            final_equity = equity_rows[-1]["equity"]
            final_equity_pct = ((final_equity - initial_equity) / initial_equity) * 100.0

            # Compute max DD for this cycle
            cycle_equity_df = pd.DataFrame(equity_rows)
            max_dd_pct = cycle_equity_df['total_dd_pct'].max()

            # Duration
            duration_days = (end_time - start_time).days if end_time else 0
        else:
            final_equity = initial_equity
            final_equity_pct = 0.0
            max_dd_pct = 0.0
            duration_days = 0

        cycle_info = {
            "cycle_id": cycle_id,
            "start_time": start_time,
            "end_time": end_time,
            "result": terminal_reason,
            "n_trades": n_trades,
            "final_equity": final_equity,
            "final_equity_pct": final_equity_pct,
            "max_drawdown_pct": max_dd_pct,
            "duration_days": duration_days,
        }
        all_cycles.append(cycle_info)

        print(f"[CYCLE {cycle_id}] Result: {terminal_reason}, Equity: ${final_equity:,.2f} ({final_equity_pct:+.2f}%), Trades: {n_trades}, Duration: {duration_days} days")

        # If single_eval mode, stop after first cycle
        if single_eval:
            print(f"\n[SINGLE-EVAL MODE] Stopping after cycle {cycle_id}")
            break

        # If incomplete (ran out of data), stop
        if terminal_reason == "INCOMPLETE":
            print(f"\n[END] Ran out of data at cycle {cycle_id}")
            break

        # Otherwise, start next cycle at next bar
        next_start_idx = end_idx + 1
        cycle_id += 1

        # Safety limit
        if cycle_id > 1000:
            print(f"\n[WARNING] Reached safety limit of 1000 cycles, stopping")
            break

    print(f"\n[COMPLETE] Ran {len(all_cycles)} cycle(s)")

    # 7) Save results
    print("\n[7/7] Saving results...")

    # Create equity curve DataFrame
    equity_df = pd.DataFrame(all_equity_rows)
    equity_df.to_csv(output_equity_csv, index=False)
    print(f"[OK] Equity curve saved to: {output_equity_csv}")

    # Save trade log
    if all_trade_logs:
        trade_log_path = output_equity_csv.replace("_equity.csv", "_trades.csv").replace(".csv", "_trades.csv")
        trade_log_df = pd.DataFrame(all_trade_logs)
        trade_log_df.to_csv(trade_log_path, index=False)
        print(f"[OK] Trade log saved to: {trade_log_path}")

    # Save cycles CSV
    if output_cycles_csv and all_cycles:
        cycles_df = pd.DataFrame(all_cycles)
        cycles_df.to_csv(output_cycles_csv, index=False)
        print(f"[OK] Cycles saved to: {output_cycles_csv}")

    # Compute summary statistics
    total_trades = len(all_trade_logs)
    max_dd_pct = equity_df['total_dd_pct'].max() if not equity_df.empty else 0.0

    # Time range
    start_time = time_grid[0]
    end_time = all_cycles[-1]["end_time"] if all_cycles and all_cycles[-1]["end_time"] else time_grid[-1]
    days = (end_time - start_time).days
    years = days / 365.25

    # Cycle statistics
    n_cycles = len(all_cycles)
    n_pass = sum(1 for c in all_cycles if c["result"] == "PASS")
    n_fail = sum(1 for c in all_cycles if c["result"].startswith("FAIL"))
    pass_rate = (n_pass / n_cycles * 100.0) if n_cycles > 0 else 0.0

    pass_cycles = [c for c in all_cycles if c["result"] == "PASS"]
    if pass_cycles:
        avg_pass_duration = float(np.mean([c["duration_days"] for c in pass_cycles]))
        median_pass_duration = float(np.median([c["duration_days"] for c in pass_cycles]))
        max_pass_duration = float(np.max([c["duration_days"] for c in pass_cycles]))
        avg_pass_equity_pct = float(np.mean([c["final_equity_pct"] for c in pass_cycles]))
    else:
        avg_pass_duration = 0.0
        median_pass_duration = 0.0
        max_pass_duration = 0.0
        avg_pass_equity_pct = 0.0

    # Convert cycles timestamps to strings for JSON serialization
    cycles_for_json = []
    for cycle in all_cycles:
        cycle_dict = cycle.copy()
        if cycle_dict["start_time"] is not None:
            cycle_dict["start_time"] = str(cycle_dict["start_time"])
        if cycle_dict["end_time"] is not None:
            cycle_dict["end_time"] = str(cycle_dict["end_time"])
        cycles_for_json.append(cycle_dict)

    summary = {
        "portfolio_id": portfolio_id,
        "mode": mode,
        "initial_equity": float(initial_equity),
        "rolling_mode": not single_eval,
        "n_cycles": int(n_cycles),
        "n_pass": int(n_pass),
        "n_fail": int(n_fail),
        "pass_rate_pct": float(pass_rate),
        "total_trades": int(total_trades),
        "max_drawdown_pct": float(max_dd_pct),
        "n_instruments": len(instrument_data),
        "simulation_days": int(days),
        "simulation_years": float(years),
        "start_date": str(start_time.date()),
        "end_date": str(end_time.date()),
        "avg_pass_duration_days": float(avg_pass_duration),
        "median_pass_duration_days": float(median_pass_duration),
        "max_pass_duration_days": float(max_pass_duration),
        "avg_pass_equity_pct": float(avg_pass_equity_pct),
        "cycles": cycles_for_json,
    }

    # Print summary
    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)
    for key, value in summary.items():
        if key == "cycles":
            continue  # Skip cycles list in console output
        elif isinstance(value, float):
            print(f"{key:30s}: {value:,.2f}")
        elif isinstance(value, bool):
            print(f"{key:30s}: {value}")
        else:
            print(f"{key:30s}: {value}")
    print("=" * 80)

    # Print cycle statistics breakdown
    if n_cycles > 1:
        print("\n" + "=" * 80)
        print("CYCLE STATISTICS")
        print("=" * 80)
        print(f"Total cycles:            {n_cycles}")
        print(f"PASS:                    {n_pass} ({pass_rate:.1f}%)")
        print(f"FAIL:                    {n_fail}")
        if pass_cycles:
            print(f"\nPASS cycles duration:")
            print(f"  Average:               {avg_pass_duration:.1f} days")
            print(f"  Median:                {median_pass_duration:.1f} days")
            print(f"  Max:                   {max_pass_duration:.0f} days")
            print(f"  Avg equity gain:       {avg_pass_equity_pct:+.2f}%")
        print("=" * 80)

    # Save summary JSON
    if output_summary_json:
        with open(output_summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Summary saved to: {output_summary_json}")

    print("\n[DONE] Simulation complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Historical live simulation for Trend_EMA_Pullback portfolio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["eval", "funded"],
        help="Trading mode: 'eval' (default) or 'funded'"
    )

    parser.add_argument(
        "--portfolio-id",
        type=str,
        default="phase2_core",
        help="Portfolio identifier (default: phase2_core)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for simulation (ISO format: YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for simulation (ISO format: YYYY-MM-DD)"
    )

    parser.add_argument(
        "--output-equity-csv",
        type=str,
        default="live_sim_equity.csv",
        help="Path to save equity curve CSV (default: live_sim_equity.csv)"
    )

    parser.add_argument(
        "--output-summary-json",
        type=str,
        default=None,
        help="Optional path to save summary statistics JSON"
    )

    parser.add_argument(
        "--output-cycles-csv",
        type=str,
        default=None,
        help="Optional path to save cycles CSV (for rolling eval mode)"
    )

    parser.add_argument(
        "--single-eval",
        action="store_true",
        help="Stop after first terminal condition (default: rolling eval mode)"
    )

    parser.add_argument(
        "--debug-risk",
        action="store_true",
        help="Enable verbose risk state logging"
    )

    args = parser.parse_args()

    # Run simulation
    run_simulation(
        mode=args.mode,
        portfolio_id=args.portfolio_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_equity_csv=args.output_equity_csv,
        output_summary_json=args.output_summary_json,
        output_cycles_csv=args.output_cycles_csv,
        single_eval=args.single_eval,
        debug_risk=args.debug_risk,
    )


if __name__ == "__main__":
    main()
