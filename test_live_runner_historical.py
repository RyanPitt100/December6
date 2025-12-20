#!/usr/bin/env python3
"""
test_live_runner_historical.py

Test the live runner's signal generation code path using historical data.

This script replays historical bars through the EXACT same functions used in
live_mt5_eval_runner.py, catching bugs that would crash the live bot.

Usage:
    python test_live_runner_historical.py --start-date 2024-01-01 --end-date 2024-03-01
    python test_live_runner_historical.py --days 30  # Last 30 days of available data
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd

from config_loader import load_all_configs
from multi_tf_builder import build_multi_tf_frame
from live_signal_generator import generate_signals_for_portfolio
from portfolio_controller import TradeSignal


def load_instrument_data(
    portfolio_cfg,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Load multi-timeframe data for all portfolio instruments."""
    instrument_data = {}

    for inst_cfg in portfolio_cfg.instruments:
        symbol = inst_cfg.symbol
        print(f"[LOAD] Loading data for {symbol}...")

        try:
            df = build_multi_tf_frame(symbol)
            df = df.sort_index()

            # Filter by date range if provided
            if start_date:
                start_dt = pd.to_datetime(start_date)
                if df.index.tz is not None:
                    start_dt = start_dt.tz_localize(df.index.tz)
                df = df[df.index >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                if df.index.tz is not None:
                    end_dt = end_dt.tz_localize(df.index.tz)
                df = df[df.index <= end_dt]

            if df.empty:
                print(f"[SKIP] {symbol}: No data after filtering")
                continue

            instrument_data[symbol] = df
            print(f"[OK] {symbol}: {len(df)} bars loaded")

        except Exception as e:
            print(f"[ERROR] Failed to load {symbol}: {e}")
            continue

    return instrument_data


def test_signal_generation_historical(
    start_date: Optional[str],
    end_date: Optional[str],
    days: Optional[int],
    sample_every_n_bars: int = 4,  # Test every Nth bar to speed up
) -> int:
    """
    Test signal generation using historical data.

    Returns:
        Number of signals found (0 if no signals, -1 if error)
    """
    print("=" * 70)
    print("HISTORICAL SIGNAL GENERATION TEST")
    print("=" * 70)
    print("This test runs the EXACT same code path as live_mt5_eval_runner.py")
    print("Any crashes here would also crash the live bot.\n")

    # Load configs
    print("[1/4] Loading configurations...")
    try:
        portfolio_cfg, risk_cfg, ftmo_cfg = load_all_configs()
        print(f"[OK] Portfolio: {portfolio_cfg.portfolio_id}")
        print(f"[OK] Instruments: {[inst.symbol for inst in portfolio_cfg.instruments]}")
    except Exception as e:
        print(f"[ERROR] Failed to load configs: {e}")
        return -1

    # Determine date range
    if days:
        end_date = None  # Use all available data
        start_date = None  # Will filter after loading

    # Load instrument data
    print("\n[2/4] Loading instrument data...")
    instrument_data = load_instrument_data(portfolio_cfg, start_date, end_date)

    if not instrument_data:
        print("[ERROR] No instrument data loaded")
        return -1

    # If using --days, filter to last N days
    if days:
        print(f"\n[2b/4] Filtering to last {days} days...")
        for symbol in instrument_data:
            df = instrument_data[symbol]
            if len(df) > 0:
                end_time = df.index[-1]
                start_time = end_time - pd.Timedelta(days=days)
                instrument_data[symbol] = df[df.index >= start_time]
                print(f"  {symbol}: {len(instrument_data[symbol])} bars")

    # Build time grid
    print("\n[3/4] Building time grid...")
    all_times = set()
    for df in instrument_data.values():
        all_times.update(df.index)
    time_grid = sorted(all_times)

    if not time_grid:
        print("[ERROR] No timestamps in data")
        return -1

    print(f"[OK] {len(time_grid)} total timestamps")
    print(f"[OK] Range: {time_grid[0]} to {time_grid[-1]}")

    # Sample bars to test (skip some for speed)
    test_times = time_grid[::sample_every_n_bars]
    print(f"[OK] Testing {len(test_times)} bars (every {sample_every_n_bars}th bar)")

    # Run signal generation
    print("\n[4/4] Running signal generation test...")
    print("-" * 70)

    portfolio_instruments = [inst.symbol for inst in portfolio_cfg.instruments]
    total_signals = 0
    errors = 0

    for i, current_ts in enumerate(test_times):
        try:
            # This is the EXACT same call as in live_mt5_eval_runner.py line 297-302
            signals = generate_signals_for_portfolio(
                instrument_data=instrument_data,
                current_ts=current_ts,
                portfolio_instruments=portfolio_instruments,
                params=None,
            )

            if signals:
                total_signals += len(signals)
                print(f"\n[SIGNAL] {current_ts}")

                # This is the EXACT same printing logic as live_mt5_eval_runner.py
                # (after the fix) - if this crashes, the live bot would too
                for sig in signals:
                    tp_str = f"{sig.tp_price:.5f}" if sig.tp_price else "None"
                    print(f"  - {sig.instrument} {sig.direction.upper()} @ "
                          f"{sig.entry_price:.5f} (SL: {sig.sl_price:.5f}, TP: {tp_str})")

            # Progress indicator
            if (i + 1) % 100 == 0:
                pct = (i + 1) / len(test_times) * 100
                print(f"[PROGRESS] {i + 1}/{len(test_times)} bars tested ({pct:.1f}%), "
                      f"{total_signals} signals found so far")

        except Exception as e:
            errors += 1
            print(f"\n[ERROR] Bar {current_ts}: {e}")
            import traceback
            traceback.print_exc()

            if errors >= 5:
                print("\n[ABORT] Too many errors, stopping test")
                return -1

    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Bars tested:     {len(test_times)}")
    print(f"Signals found:   {total_signals}")
    print(f"Errors:          {errors}")

    if errors == 0:
        print(f"\n[PASS] Signal generation code path works correctly")
        print(f"       Found {total_signals} signals over {len(test_times)} bars")
        if total_signals == 0:
            print(f"       (No signals is OK - regime conditions may not have been met)")
    else:
        print(f"\n[FAIL] {errors} error(s) occurred - these would crash the live bot")

    print("=" * 70)

    return total_signals if errors == 0 else -1


def main():
    parser = argparse.ArgumentParser(
        description="Test live runner signal generation using historical data",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Test last N days of available data"
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=4,
        help="Test every Nth bar (default: 4, meaning every hour for 15m bars)"
    )

    args = parser.parse_args()

    # Default to last 60 days if no range specified
    if not args.start_date and not args.end_date and not args.days:
        args.days = 60
        print(f"[INFO] No date range specified, using last {args.days} days\n")

    result = test_signal_generation_historical(
        start_date=args.start_date,
        end_date=args.end_date,
        days=args.days,
        sample_every_n_bars=args.sample_every,
    )

    sys.exit(0 if result >= 0 else 1)


if __name__ == "__main__":
    main()
