#!/usr/bin/env python3
"""
view_logs.py

Simple CLI tool to view trading bot logs from the SQLite database.

Usage:
    python view_logs.py                    # Show recent activity summary
    python view_logs.py --signals          # Show recent signals
    python view_logs.py --account          # Show account snapshots
    python view_logs.py --events           # Show system events
    python view_logs.py --errors           # Show errors only
    python view_logs.py --stats            # Show trade statistics
"""

import argparse
import sys
from datetime import datetime
from trade_logger import get_logger


def format_timestamp(ts_str: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ts_str


def show_summary(logger):
    """Show recent activity summary."""
    print("=" * 80)
    print("TRADING BOT ACTIVITY SUMMARY")
    print("=" * 80)

    # Recent account snapshot
    snapshots = logger.get_recent_account_snapshots(limit=1)
    if snapshots:
        snap = snapshots[0]
        print(f"\nLatest Account State ({format_timestamp(snap['timestamp'])}):")
        print(f"  Balance:     ${snap['balance']:,.2f}")
        print(f"  Equity:      ${snap['equity']:,.2f}")
        print(f"  Daily P&L:   {snap['daily_pnl_pct']:.2f}%")
        print(f"  Total DD:    {snap['total_dd_pct']:.2f}%")
        print(f"  Open Pos:    {snap['open_positions']}")
        print(f"  Risk State:  {snap['risk_state']}")
        print(f"  Risk Action: {snap['risk_action']}")
    else:
        print("\nNo account snapshots logged yet")

    # Recent signals
    signals = logger.get_recent_signals(limit=5)
    print(f"\nRecent Signals ({len(signals)}):")
    if signals:
        for sig in signals:
            executed = "✓ EXECUTED" if sig['executed'] else "✗ Not executed"
            print(f"  {format_timestamp(sig['timestamp'])} | {sig['instrument']} {sig['direction'].upper()} | {executed}")
    else:
        print("  No signals logged yet")

    # Open executions (current trades)
    open_trades = logger.get_open_executions()
    print(f"\nOpen Trades ({len(open_trades)}):")
    if open_trades:
        for trade in open_trades:
            print(f"  {trade['instrument']} {trade['direction'].upper()} @ {trade['entry_price']:.5f}")
            print(f"    SL: {trade['sl_price']:.5f} | Size: {trade['size_lots']:.2f} lots")
    else:
        print("  No open trades")

    # Recent events
    events = logger.get_recent_events(limit=5)
    print(f"\nRecent Events ({len(events)}):")
    if events:
        for evt in events:
            level_icon = {"INFO": "ℹ", "WARNING": "⚠", "ERROR": "✗"}.get(evt['level'], "•")
            print(f"  {level_icon} {format_timestamp(evt['timestamp'])} | [{evt['category']}] {evt['message']}")
    else:
        print("  No events logged yet")

    # Trade statistics
    stats = logger.get_trade_statistics()
    if stats['total_trades'] > 0:
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Wins/Losses:  {stats['wins']}/{stats['losses']}")
        print(f"  Win Rate:     {stats['win_rate']:.1f}%")
        print(f"  Avg Win:      ${stats['avg_win']:,.2f}")
        print(f"  Avg Loss:     ${stats['avg_loss']:,.2f}")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Total P&L:    ${stats['total_pnl']:,.2f}")
        print(f"  Avg R:        {stats['avg_r_multiple']:.2f}R")
    else:
        print(f"\nNo closed trades yet")


def show_signals(logger, limit=20):
    """Show recent signals."""
    print("=" * 80)
    print(f"RECENT SIGNALS (last {limit})")
    print("=" * 80)

    signals = logger.get_recent_signals(limit=limit)

    if not signals:
        print("No signals logged yet")
        return

    for sig in signals:
        executed = "✓ EXECUTED" if sig['executed'] else "✗ Not executed"
        print(f"\n{format_timestamp(sig['timestamp'])} | {sig['instrument']} {sig['direction'].upper()}")
        tp_str = f"{sig['tp_price']:.5f}" if sig['tp_price'] else "None"
        print(f"  Entry: {sig['entry_price']:.5f} | SL: {sig['sl_price']:.5f} | TP: {tp_str}")
        print(f"  Reason: {sig['reason']}")
        if sig['regime_h1']:
            print(f"  Regime: H1={sig['regime_h1']}, H4={sig['regime_h4']}, D1={sig['regime_d1']}")
        print(f"  Status: {executed}")


def show_account(logger, limit=20):
    """Show account snapshots."""
    print("=" * 80)
    print(f"ACCOUNT SNAPSHOTS (last {limit})")
    print("=" * 80)

    snapshots = logger.get_recent_account_snapshots(limit=limit)

    if not snapshots:
        print("No account snapshots logged yet")
        return

    for snap in snapshots:
        print(f"\n{format_timestamp(snap['timestamp'])}")
        print(f"  Balance:     ${snap['balance']:,.2f}")
        print(f"  Equity:      ${snap['equity']:,.2f}")
        print(f"  Daily P&L:   {snap['daily_pnl_pct']:+.2f}% (${snap['realised_pnl_today']:+,.2f})")
        print(f"  Total DD:    {snap['total_dd_pct']:.2f}%")
        print(f"  Open Pos:    {snap['open_positions']}")
        print(f"  Risk State:  {snap['risk_state']}")
        print(f"  Risk Action: {snap['risk_action']}")

        flags = []
        if snap['eval_profit_target_hit']:
            flags.append("EVAL_TARGET_HIT")
        if snap['breached_daily_limit']:
            flags.append("DAILY_LIMIT_BREACHED")
        if snap['breached_total_limit']:
            flags.append("TOTAL_LIMIT_BREACHED")

        if flags:
            print(f"  Flags: {', '.join(flags)}")


def show_events(logger, limit=50, level=None):
    """Show system events."""
    title = f"SYSTEM EVENTS"
    if level:
        title += f" ({level} only)"
    title += f" (last {limit})"

    print("=" * 80)
    print(title)
    print("=" * 80)

    events = logger.get_recent_events(level=level, limit=limit)

    if not events:
        print("No events logged yet")
        return

    for evt in events:
        level_icon = {"INFO": "ℹ", "WARNING": "⚠", "ERROR": "✗"}.get(evt['level'], "•")
        print(f"\n{level_icon} {format_timestamp(evt['timestamp'])} | [{evt['category']}] {evt['level']}")
        print(f"  {evt['message']}")
        if evt['details']:
            print(f"  Details: {evt['details'][:200]}{'...' if len(evt['details']) > 200 else ''}")


def show_stats(logger):
    """Show trade statistics."""
    print("=" * 80)
    print("TRADE STATISTICS")
    print("=" * 80)

    stats = logger.get_trade_statistics()

    if stats['total_trades'] == 0:
        print("\nNo closed trades yet")
        return

    print(f"\nTotal Trades:     {stats['total_trades']}")
    print(f"Wins:             {stats['wins']}")
    print(f"Losses:           {stats['losses']}")
    print(f"Win Rate:         {stats['win_rate']:.1f}%")
    print(f"\nAverage Win:      ${stats['avg_win']:,.2f}")
    print(f"Average Loss:     ${stats['avg_loss']:,.2f}")
    print(f"Profit Factor:    {stats['profit_factor']:.2f}")
    print(f"\nTotal P&L:        ${stats['total_pnl']:+,.2f}")
    print(f"Average R:        {stats['avg_r_multiple']:.2f}R")

    # Show closed trades
    closed = logger.get_closed_trades(limit=10)
    if closed:
        print(f"\n{'=' * 80}")
        print(f"RECENT CLOSED TRADES (last 10)")
        print(f"{'=' * 80}")

        for trade in closed:
            pnl_sign = "+" if trade['pnl_currency'] > 0 else ""
            r_str = f"{trade['r_multiple']:.2f}R" if trade['r_multiple'] else "N/A"
            print(f"\n{format_timestamp(trade['timestamp'])} | {trade['instrument']} {trade['direction'].upper()}")
            print(f"  Entry: {trade['entry_price']:.5f} → Exit: {trade['exit_price']:.5f}")
            print(f"  P&L: {pnl_sign}${trade['pnl_currency']:,.2f} ({r_str})")
            print(f"  Exit: {trade['exit_reason']} | Duration: {trade['duration_minutes']} min")


def main():
    parser = argparse.ArgumentParser(description="View trading bot logs")
    parser.add_argument("--signals", action="store_true", help="Show recent signals")
    parser.add_argument("--account", action="store_true", help="Show account snapshots")
    parser.add_argument("--events", action="store_true", help="Show system events")
    parser.add_argument("--errors", action="store_true", help="Show errors only")
    parser.add_argument("--stats", action="store_true", help="Show trade statistics")
    parser.add_argument("--limit", type=int, default=20, help="Number of records to show (default: 20)")
    parser.add_argument("--db", type=str, default="trades.db", help="Database file path (default: trades.db)")

    args = parser.parse_args()

    # Initialize logger
    logger = get_logger(args.db)

    # Determine what to show
    if args.signals:
        show_signals(logger, limit=args.limit)
    elif args.account:
        show_account(logger, limit=args.limit)
    elif args.events:
        show_events(logger, limit=args.limit)
    elif args.errors:
        show_events(logger, limit=args.limit, level="ERROR")
    elif args.stats:
        show_stats(logger)
    else:
        # Default: show summary
        show_summary(logger)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
