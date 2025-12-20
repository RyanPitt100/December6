#!/usr/bin/env python3
"""
live_mt5_eval_runner.py

Live MetaTrader 5 paper trading runner for FTMO eval mode.

This script connects to MT5 and runs the live trading engine using the existing
risk management and portfolio controller logic. It operates as a real-time version
of live_sim_trend_portfolio.py but with actual MT5 data instead of historical bars.

Architecture:
    - MT5 Bridge (mt5_bridge.py): Handles MT5 I/O (account state, prices, orders)
    - Risk Manager (risk_manager.py): FTMO overlays, daily/total limits, envelope guard
    - Portfolio Controller (portfolio_controller.py): Order generation, position sizing

Usage:
    python live_mt5_eval_runner.py --mode eval --poll-interval-seconds 60

Arguments:
    --mode: Trading mode ("eval" or "funded")
    --poll-interval-seconds: Loop frequency in seconds (default: 60)
    --max-iterations: Max loop iterations (default: unlimited, use for testing)
"""

import argparse
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional
from functools import wraps

# === EARLY INIT: Print immediately to confirm script is running ===
print("=" * 60, flush=True)
print("LIVE MT5 EVAL RUNNER - INITIALIZING", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Time: {datetime.now()}", flush=True)
print("=" * 60, flush=True)

# Force unbuffered output so prints show immediately
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

print("[INIT] Importing pandas...", flush=True)
import pandas as pd
print("[INIT] Pandas OK", flush=True)


# ---------------------------------------------------------------------------
# Timeout Decorator (prevents data fetch from hanging indefinitely)
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def timeout(seconds):
    """Decorator that raises TimeoutError if function takes too long."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Windows doesn't support signal.SIGALRM, use threading approach
            import threading
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator

print("[INIT] Importing config_loader...", flush=True)
from config_loader import load_all_configs, load_discord_config
print("[INIT] Importing risk_manager...", flush=True)
from risk_manager import evaluate_daily_risk_state, describe_risk_state
print("[INIT] Importing portfolio_controller...", flush=True)
from portfolio_controller import TradeSignal, decide_portfolio_orders, describe_orders
print("[INIT] Importing mt5_bridge...", flush=True)
import mt5_bridge
print("[INIT] Importing mt5_multi_tf_builder...", flush=True)
from mt5_multi_tf_builder import build_multi_tf_from_mt5, get_latest_completed_bar_time
print("[INIT] Importing live_signal_generator...", flush=True)
from live_signal_generator import generate_signals_for_portfolio
print("[INIT] Importing trade_logger...", flush=True)
from trade_logger import get_logger
print("[INIT] Importing discord_notifier...", flush=True)
from discord_notifier import get_discord_notifier
print("[INIT] Importing yaml...", flush=True)
import yaml
print("[INIT] All imports complete!", flush=True)


def load_prop_firm_from_live_config() -> str:
    """Load prop firm selection from live_config.yml."""
    try:
        with open("live_config.yml", "r") as f:
            config = yaml.safe_load(f)
        prop_firm = config.get("account", {}).get("prop_firm", "ftmo")
        return prop_firm.lower()
    except Exception as e:
        print(f"[WARNING] Could not load prop_firm from live_config.yml: {e}")
        return "ftmo"


# ---------------------------------------------------------------------------
# Signal Generation (Real Implementation)
# ---------------------------------------------------------------------------

# Cache for instrument data to avoid refetching on every iteration
_instrument_data_cache: Dict[str, pd.DataFrame] = {}
_last_full_refresh: Optional[datetime] = None
_FULL_REFRESH_INTERVAL_HOURS = 4  # Full refresh every 4 hours
_DATA_FETCH_TIMEOUT_SECONDS = 30  # Max time to fetch data for one instrument


@timeout(_DATA_FETCH_TIMEOUT_SECONDS)
def _fetch_instrument_data_full(symbol: str, n_bars: int = 500):
    """Fetch full multi-TF data with timeout protection."""
    return build_multi_tf_from_mt5(symbol, n_bars_15m=n_bars)


@timeout(_DATA_FETCH_TIMEOUT_SECONDS)
def _fetch_instrument_data_incremental(symbol: str, n_bars: int = 20):
    """Fetch recent bars only for incremental update."""
    return build_multi_tf_from_mt5(symbol, n_bars_15m=n_bars)


def generate_signals(portfolio_cfg) -> List[TradeSignal]:
    """
    Generate trading signals using the canonical TrendEMAPullback strategy.

    Uses smart caching:
    - Full refresh (500 bars) on startup and every 4 hours
    - Incremental updates (20 bars) every 15 minutes in between

    Args:
        portfolio_cfg: Portfolio configuration with list of instruments

    Returns:
        List of TradeSignal objects ready for portfolio controller
    """
    global _instrument_data_cache, _last_full_refresh

    current_time = pd.Timestamp.now(tz='UTC')

    # Determine if we need a full refresh or just incremental
    needs_full_refresh = (
        _last_full_refresh is None
        or len(_instrument_data_cache) == 0
        or (current_time - _last_full_refresh).total_seconds() > _FULL_REFRESH_INTERVAL_HOURS * 3600
    )

    if needs_full_refresh:
        # Full refresh - fetch all 500 bars
        print(f"[SIGNAL GEN] Full data refresh (every {_FULL_REFRESH_INTERVAL_HOURS}h)...")
        _instrument_data_cache.clear()

        for inst_cfg in portfolio_cfg.instruments:
            symbol = inst_cfg.symbol
            print(f"[SIGNAL GEN] Fetching full data for {symbol}...")

            try:
                df = _fetch_instrument_data_full(symbol, n_bars=500)

                if df is None or df.empty:
                    print(f"[SIGNAL GEN] Warning: No data for {symbol}")
                    continue

                required_cols = ['close', 'regime_h1', 'regime_h4']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"[SIGNAL GEN] Warning: {symbol} missing columns: {missing_cols}")
                    continue

                _instrument_data_cache[symbol] = df
                print(f"[SIGNAL GEN] OK: {symbol} loaded {len(df)} bars")

            except TimeoutError:
                print(f"[SIGNAL GEN] TIMEOUT loading {symbol} (>{_DATA_FETCH_TIMEOUT_SECONDS}s) - skipping")
                continue
            except Exception as e:
                print(f"[SIGNAL GEN] Error loading {symbol}: {e}")
                continue

        _last_full_refresh = current_time
        print(f"[SIGNAL GEN] Full refresh complete: {len(_instrument_data_cache)} instruments")

    else:
        # Incremental update - fetch only recent bars and update cache
        print(f"[SIGNAL GEN] Incremental update (20 bars)...")

        for inst_cfg in portfolio_cfg.instruments:
            symbol = inst_cfg.symbol

            if symbol not in _instrument_data_cache:
                print(f"[SIGNAL GEN] {symbol} not in cache, fetching full...")
                try:
                    df = _fetch_instrument_data_full(symbol, n_bars=500)
                    if df is not None and not df.empty:
                        _instrument_data_cache[symbol] = df
                        print(f"[SIGNAL GEN] OK: {symbol} loaded {len(df)} bars")
                except Exception as e:
                    print(f"[SIGNAL GEN] Error loading {symbol}: {e}")
                continue

            try:
                # Fetch recent bars
                df_recent = _fetch_instrument_data_incremental(symbol, n_bars=20)

                if df_recent is None or df_recent.empty:
                    print(f"[SIGNAL GEN] Warning: No recent data for {symbol}")
                    continue

                # Update cache: keep old bars, replace/add recent ones
                df_cached = _instrument_data_cache[symbol]

                # Remove bars that will be replaced
                cutoff_time = df_recent.index.min()
                df_old = df_cached[df_cached.index < cutoff_time]

                # Combine old + new
                df_updated = pd.concat([df_old, df_recent]).sort_index()

                # Keep only last 500 bars to prevent memory growth
                if len(df_updated) > 500:
                    df_updated = df_updated.tail(500)

                _instrument_data_cache[symbol] = df_updated

            except TimeoutError:
                print(f"[SIGNAL GEN] TIMEOUT updating {symbol} - using cached data")
                continue
            except Exception as e:
                print(f"[SIGNAL GEN] Error updating {symbol}: {e} - using cached data")
                continue

        print(f"[SIGNAL GEN] Incremental update complete")

    # If no data loaded, return empty
    if not _instrument_data_cache:
        print(f"[SIGNAL GEN] No instrument data available")
        return []

    # Determine latest completed bar time
    # Use first instrument's data to find the bar boundary
    first_symbol = list(_instrument_data_cache.keys())[0]
    first_df = _instrument_data_cache[first_symbol]

    latest_bar_time = get_latest_completed_bar_time(first_df, current_time)

    if latest_bar_time is None:
        print(f"[SIGNAL GEN] No completed bars available yet")
        return []

    print(f"[SIGNAL GEN] Generating signals for bar at {latest_bar_time}")

    # Log current regime for each instrument
    print(f"\n[REGIMES] Current regime state:")
    for symbol, df in _instrument_data_cache.items():
        try:
            # Get regime at latest bar time
            if latest_bar_time in df.index:
                row = df.loc[latest_bar_time]
            else:
                # Find closest bar before latest_bar_time
                valid_idx = df.index[df.index <= latest_bar_time]
                if len(valid_idx) == 0:
                    print(f"  {symbol}: NO DATA")
                    continue
                row = df.loc[valid_idx[-1]]

            regime_h1 = row.get('regime_h1', 'N/A')
            regime_h4 = row.get('regime_h4', 'N/A')
            regime_d1 = row.get('regime_d1', 'N/A')
            print(f"  {symbol}: H1={regime_h1:<12} H4={regime_h4:<12} D1={regime_d1}")
        except Exception as e:
            print(f"  {symbol}: Error reading regime - {e}")

    # Generate signals for all portfolio instruments
    portfolio_instruments = [inst.symbol for inst in portfolio_cfg.instruments]

    signals = generate_signals_for_portfolio(
        instrument_data=_instrument_data_cache,
        current_ts=latest_bar_time,
        portfolio_instruments=portfolio_instruments,
        params=None,  # Use default TrendEMAPullbackParams
    )

    if signals:
        print(f"[SIGNAL GEN] Generated {len(signals)} signal(s):")
        for sig in signals:
            tp_str = f"{sig.tp_price:.5f}" if sig.tp_price else "None"
            print(f"  - {sig.instrument} {sig.direction.upper()} @ {sig.entry_price:.5f} (SL: {sig.sl_price:.5f}, TP: {tp_str})")
    else:
        print(f"[SIGNAL GEN] No signals at {latest_bar_time}")

    return signals


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def run_live_loop(
    mode: str,
    poll_interval_seconds: int,
    max_iterations: int,
    dry_run: bool = False,
) -> None:
    """
    Main live trading loop.

    Args:
        mode: Trading mode ("eval" or "funded")
        poll_interval_seconds: Sleep duration between iterations
        max_iterations: Max iterations (0 = unlimited)
        dry_run: If True, generate signals and orders but do NOT execute on MT5
    """
    print("=" * 80)
    print("LIVE MT5 EVAL RUNNER")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Poll interval: {poll_interval_seconds}s")
    print(f"Dry-run: {dry_run}")
    print("=" * 80)

    # 0) Initialize logger and Discord
    logger = get_logger("trades.db")
    logger.log_event("INFO", "bot_startup", f"Bot starting in {mode} mode (dry_run={dry_run})")
    print("\n[0/5] Trade logger initialized (trades.db)")

    # Initialize Discord notifier (optional)
    discord_config = load_discord_config()
    discord = get_discord_notifier(discord_config)
    if discord:
        print("[0/5] Discord alerts enabled")
        discord.notify_bot_status("startup", f"Bot starting in {mode} mode (dry_run={dry_run})")
    else:
        print("[0/5] Discord alerts disabled")

    # 1) Initialize MT5
    print("\n[1/4] Initializing MT5 connection...")
    try:
        mt5_bridge.init_mt5()
        logger.log_event("INFO", "mt5_connection", "MT5 connected successfully")
    except Exception as e:
        print(f"[ERROR] MT5 initialization failed: {e}")
        logger.log_event("ERROR", "mt5_connection", f"MT5 initialization failed: {e}")
        sys.exit(1)

    # 2) Load configurations
    print("\n[2/4] Loading configurations...")
    prop_firm = load_prop_firm_from_live_config()
    print(f"[OK] Prop firm: {prop_firm.upper()}")
    try:
        portfolio_cfg, risk_cfg, ftmo_cfg = load_all_configs(prop_firm=prop_firm)
    except Exception as e:
        print(f"[ERROR] Failed to load configs: {e}")
        mt5_bridge.shutdown_mt5()
        sys.exit(1)

    print(f"[OK] Portfolio: {portfolio_cfg.portfolio_id}")
    print(f"[OK] Instruments: {[inst.symbol for inst in portfolio_cfg.instruments]}")

    # Get risk per trade for this mode
    if mode == "eval":
        risk_per_trade = risk_cfg.risk_per_trade_eval_pct
    else:
        risk_per_trade = risk_cfg.risk_per_trade_funded_pct

    print(f"[OK] Risk per trade ({mode}): {risk_per_trade}%")

    # Print prop firm limits
    print(f"[OK] {ftmo_cfg.account_type} limits:")
    print(f"     Daily loss limit: {ftmo_cfg.limits.internal_daily_loss_limit_pct}%")
    print(f"     Total loss limit: {ftmo_cfg.limits.internal_total_loss_limit_pct}%")
    if ftmo_cfg.limits.internal_max_unrealised_loss_pct > 0:
        print(f"     Max unrealised loss: {ftmo_cfg.limits.internal_max_unrealised_loss_pct}%")
    if mode == "eval" and ftmo_cfg.eval_phase and ftmo_cfg.eval_phase.enabled:
        print(f"     Profit target: {ftmo_cfg.eval_phase.profit_target_pct}%")
        print(f"     Lock trading on target: {ftmo_cfg.eval_phase.lock_trading_on_target}")

    # 3) Get instrument symbols
    print("\n[3/4] Preparing instrument list...")
    instruments = [inst.symbol for inst in portfolio_cfg.instruments]
    print(f"[OK] Monitoring {len(instruments)} instruments: {instruments}")

    # 4) Start main loop
    print("\n[4/4] Starting live loop...")
    print("=" * 80)

    iteration = 0
    terminal_condition_hit = False

    # Daily summary tracking
    trades_today = 0
    last_summary_date = None
    starting_balance = None  # Will be set on first iteration

    # Parse daily summary time from config
    if discord_config and discord_config.alerts.daily_summary:
        summary_time_str = discord_config.alerts.daily_summary_time
        summary_hour, summary_minute = map(int, summary_time_str.split(':'))
        print(f"[OK] Daily summary scheduled at {summary_time_str}")
    else:
        summary_hour, summary_minute = 17, 0  # Default 5pm

    try:
        while True:
            iteration += 1

            # Check max iterations
            if max_iterations > 0 and iteration > max_iterations:
                print(f"\n[STOP] Reached max iterations ({max_iterations})")
                break

            # Check terminal condition from previous iteration
            if terminal_condition_hit:
                print(f"\n[STOP] Terminal condition hit, stopping trading")
                break

            print(f"\n{'-' * 80}")
            print(f"[ITERATION {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'-' * 80}")

            # Step 1: Fetch current account state from MT5
            try:
                account_state = mt5_bridge.fetch_account_state(
                    portfolio_cfg, risk_cfg, ftmo_cfg
                )
            except Exception as e:
                print(f"[ERROR] Failed to fetch account state: {e}")
                logger.log_event("ERROR", "account_state", f"Failed to fetch account state: {e}")
                print(f"[ERROR] Retrying in {poll_interval_seconds}s...")
                time.sleep(poll_interval_seconds)
                continue

            # Track starting balance for daily P&L calculation
            today = datetime.now().date()
            if starting_balance is None or last_summary_date != today:
                starting_balance = account_state.balance
                if last_summary_date != today:
                    trades_today = 0  # Reset trade count for new day

            # Print account summary
            print(f"\nAccount State:")
            print(f"  Equity:      ${account_state.equity:,.2f}")
            print(f"  Balance:     ${account_state.balance:,.2f}")
            print(f"  Open positions: {len(account_state.open_positions)}")
            if account_state.open_positions:
                for pos in account_state.open_positions:
                    print(f"    - {pos.instrument} {pos.direction.upper()} @ {pos.entry_price:.5f} (SL: {pos.sl_price:.5f})")

            # Step 2: Fetch current prices (for display and future signal generation)
            try:
                prices = mt5_bridge.fetch_mid_prices(instruments)
                print(f"\nCurrent Prices:")
                for symbol, price in sorted(prices.items()):
                    print(f"  {symbol}: {price:.5f}")
            except Exception as e:
                print(f"[WARNING] Failed to fetch prices: {e}")
                prices = {}

            # Step 3: Evaluate risk state
            risk_state = evaluate_daily_risk_state(
                account_state, ftmo_cfg, mode=mode
            )

            # Log account snapshot
            logger.log_account_snapshot(account_state, risk_state)

            print(f"\nRisk State:")
            print(f"  DD today:        {risk_state['dd_today_pct']:.2f}%")
            print(f"  Open risk:       {risk_state['open_risk_pct']:.2f}%")
            print(f"  Combined risk:   {risk_state['combined_risk_pct']:.2f}%")
            print(f"  Total DD:        {risk_state['total_dd_pct']:.2f}%")
            print(f"  Eval target hit: {risk_state.get('hit_eval_profit_target', False)}")
            print(f"  Must flatten:    {risk_state.get('must_flatten', False)}")
            print(f"  Block new:       {risk_state.get('block_new_trades', False)}")

            # Send risk warnings to Discord (using config-based thresholds)
            if discord:
                daily_limit = ftmo_cfg.limits.internal_daily_loss_limit_pct
                dd_today = risk_state['dd_today_pct']
                warning_threshold = daily_limit * 0.5  # 50% of limit
                urgent_threshold = daily_limit * 0.75  # 75% of limit

                if dd_today > warning_threshold and dd_today <= urgent_threshold:
                    discord.notify_risk_warning("warning",
                        f"Daily drawdown at {dd_today:.2f}%",
                        f"Daily loss limit is {daily_limit}%. Current: {dd_today:.2f}%")
                elif dd_today > urgent_threshold and dd_today < daily_limit:
                    discord.notify_risk_warning("urgent",
                        f"Daily drawdown at {dd_today:.2f}% - Approaching limit!",
                        f"Daily loss limit is {daily_limit}%. Only {daily_limit - dd_today:.2f}% remaining!")

                # Unrealized loss warnings (Blueguardian)
                unrealised_limit = ftmo_cfg.limits.internal_max_unrealised_loss_pct
                if unrealised_limit > 0:
                    unrealised_pct = risk_state.get('unrealised_loss_pct', 0.0)
                    if unrealised_pct > unrealised_limit * 0.5 and unrealised_pct < unrealised_limit:
                        discord.notify_risk_warning("warning",
                            f"Unrealised loss at {unrealised_pct:.2f}%",
                            f"Max unrealised limit is {unrealised_limit}%. Current: {unrealised_pct:.2f}%")

            # Print detailed risk description if conditions triggered
            if (
                risk_state.get('breached_daily_limit', False)
                or risk_state.get('breached_total_limit', False)
                or risk_state.get('breached_unrealised_limit', False)
                or risk_state.get('envelope_guard_triggered', False)
                or risk_state.get('hit_eval_profit_target', False)
                or risk_state.get('hard_brake_triggered', False)
            ):
                print(f"\n{describe_risk_state(risk_state)}")

            # Step 4: Generate trading signals using canonical TrendEMAPullback strategy
            try:
                signals = generate_signals(portfolio_cfg)

                if signals:
                    print(f"\nGenerated {len(signals)} signal(s):")
                    for sig in signals:
                        print(f"  - {sig.instrument} {sig.direction.upper()} (reason: {sig.reason})")
                        # Log signal (will update with execution_id later if executed)
                        logger.log_signal(sig, regime_labels=None, executed=False)
            except Exception as e:
                print(f"[ERROR] Signal generation failed: {e}")
                logger.log_event("ERROR", "signal_generation", f"Signal generation failed: {e}", details=str(e))
                signals = []

            # NOTE: Future global risk scalar λ would be applied here
            # Example (commented out):
            # if should_scale_risk:
            #     scaled_risk_pct = risk_per_trade * lambda_scalar
            #     # Apply to signals or pass to portfolio controller
            # TODO: Implement risk scaling based on recent performance

            # Step 5: Generate orders via portfolio controller
            orders = decide_portfolio_orders(
                account_state=account_state,
                signals=signals,
                portfolio_cfg=portfolio_cfg,
                risk_cfg=risk_cfg,
                ftmo_cfg=ftmo_cfg,
                mode=mode,
            )

            if orders:
                print(f"\nPortfolio controller generated {len(orders)} order(s):")
                print(describe_orders(orders))

            # Step 6: Execute orders on MT5 (or log if dry-run)
            if orders:
                if dry_run:
                    # Dry-run mode: log orders but don't execute
                    print(f"\n[DRY-RUN] Would execute {len(orders)} order(s):")
                    for order in orders:
                        # Log order with dry-run status
                        logger.log_order(order, status="dry_run")

                        direction_str = order.direction.upper() if order.direction else ''
                        action_str = f"{order.action.upper()} {direction_str}"
                        if order.action == "open":
                            entry_str = f"{order.entry_price:.5f}" if order.entry_price else "MARKET"
                            sl_str = f"{order.sl_price:.5f}" if order.sl_price else "N/A"
                            tp_str = f"{order.tp_price:.5f}" if order.tp_price else "N/A"
                            print(f"[DRY-RUN]   {action_str} {order.instrument} | "
                                  f"Entry: {entry_str} | SL: {sl_str} | TP: {tp_str} | "
                                  f"Size: {order.size_lots:.2f} lots | Reason: {order.reason}")
                        else:
                            print(f"[DRY-RUN]   CLOSE {order.instrument} | Reason: {order.reason}")
                else:
                    # Live mode: actually execute
                    try:
                        # Log orders as pending
                        for order in orders:
                            order_id = logger.log_order(order, status="pending")

                            # Store order_id for later execution logging
                            if not hasattr(order, '_log_id'):
                                order._log_id = order_id

                        # Execute orders
                        mt5_bridge.send_orders(orders)
                        print(f"[OK] Orders executed")
                        logger.log_event("INFO", "order_execution", f"Successfully executed {len(orders)} order(s)")

                        # Track trades for daily summary
                        trades_today += sum(1 for o in orders if o.action == "open")

                        # Update orders status to filled and log executions
                        for order in orders:
                            if hasattr(order, '_log_id'):
                                # Update order status (we'd need to modify log_order to support updates)
                                # For now, just log execution
                                pass

                            # Log execution for OPEN orders
                            if order.action == "open":
                                execution_id = logger.log_execution(
                                    order_id=getattr(order, '_log_id', None),
                                    instrument=order.instrument,
                                    direction=order.direction,
                                    entry_price=order.entry_price,
                                    sl_price=order.sl_price,
                                    tp_price=order.tp_price,
                                    size_lots=order.size_lots,
                                    reason=order.reason,
                                )
                                print(f"[LOG] Execution logged: ID={execution_id}")

                                # Send Discord notification
                                if discord:
                                    discord.notify_trade_opened(
                                        instrument=order.instrument,
                                        direction=order.direction,
                                        entry=order.entry_price,
                                        sl=order.sl_price,
                                        tp=order.tp_price,
                                        size_lots=order.size_lots,
                                        reason=order.reason
                                    )

                    except Exception as e:
                        print(f"[ERROR] Failed to execute orders: {e}")
                        import traceback
                        error_details = traceback.format_exc()
                        logger.log_event("ERROR", "order_execution", f"Failed to execute orders: {e}", details=error_details)

                        # Send Discord error notification
                        if discord:
                            discord.notify_error("order_execution", f"Failed to execute orders: {e}", details=error_details[:500])

                        # Log failed orders
                        for order in orders:
                            if hasattr(order, '_log_id'):
                                # Would update order status to 'error' here
                                pass

            # Step 7: Check for terminal conditions
            if risk_state.get('must_flatten', False):
                print(f"\n[TERMINAL] Terminal condition detected")

                # Determine reason and send Discord notification
                if risk_state.get('hit_eval_profit_target', False):
                    reason = "EVAL_PROFIT_TARGET_HIT"
                    print(f"[TERMINAL] Reason: Eval profit target hit (PASS)")
                    logger.log_event("INFO", "terminal_condition", "Eval profit target hit - CHALLENGE PASSED!")
                    if discord:
                        discord.notify_risk_warning("critical",
                            "🎉 EVAL PROFIT TARGET HIT - CHALLENGE PASSED!",
                            f"Balance: ${account_state.balance:,.2f}\nAll positions will be closed.")
                elif risk_state.get('breached_total_limit', False):
                    reason = "BREACHED_TOTAL_LIMIT"
                    print(f"[TERMINAL] Reason: Total loss limit breached (FAIL)")
                    logger.log_event("ERROR", "terminal_condition", "Total loss limit breached - CHALLENGE FAILED")
                    if discord:
                        discord.notify_risk_warning("critical",
                            "🛑 TOTAL LOSS LIMIT BREACHED - CHALLENGE FAILED",
                            f"Total DD: {risk_state['total_dd_pct']:.2f}% (limit: {ftmo_cfg.limits.internal_total_loss_limit_pct}%)")
                elif risk_state.get('breached_daily_limit', False):
                    reason = "BREACHED_DAILY_LIMIT"
                    print(f"[TERMINAL] Reason: Daily loss limit breached (FAIL)")
                    logger.log_event("ERROR", "terminal_condition", "Daily loss limit breached - CHALLENGE FAILED")
                    if discord:
                        discord.notify_risk_warning("critical",
                            "🛑 DAILY LOSS LIMIT BREACHED - CHALLENGE FAILED",
                            f"Daily DD: {risk_state['dd_today_pct']:.2f}% (limit: {ftmo_cfg.limits.internal_daily_loss_limit_pct}%)")
                elif risk_state.get('breached_unrealised_limit', False):
                    reason = "BREACHED_UNREALISED_LIMIT"
                    print(f"[TERMINAL] Reason: Unrealised loss limit breached (FAIL)")
                    logger.log_event("ERROR", "terminal_condition", "Unrealised loss limit breached - CHALLENGE FAILED")
                    if discord:
                        discord.notify_risk_warning("critical",
                            "🛑 UNREALISED LOSS LIMIT BREACHED - CHALLENGE FAILED",
                            f"Unrealised loss: {risk_state.get('unrealised_loss_pct', 0):.2f}% (limit: {ftmo_cfg.limits.internal_max_unrealised_loss_pct}%)")
                elif risk_state.get('envelope_guard_triggered', False):
                    reason = "ENVELOPE_GUARD_TRIGGERED"
                    print(f"[TERMINAL] Reason: Risk envelope guard triggered (FAIL)")
                    logger.log_event("ERROR", "terminal_condition", "Risk envelope guard triggered - CHALLENGE FAILED")
                    if discord:
                        discord.notify_risk_warning("critical",
                            "🛑 RISK ENVELOPE GUARD TRIGGERED",
                            "Maximum risk envelope exceeded")
                elif risk_state.get('hard_brake_triggered', False):
                    reason = "HARD_BRAKE_TRIGGERED"
                    print(f"[TERMINAL] Reason: Hard brake triggered (FAIL)")
                    logger.log_event("ERROR", "terminal_condition", "Hard brake triggered - CHALLENGE FAILED")
                    if discord:
                        discord.notify_risk_warning("critical",
                            "🛑 HARD BRAKE TRIGGERED",
                            "Trading halted due to hard brake condition")
                else:
                    reason = "OTHER_RISK_HARD_STOP"
                    print(f"[TERMINAL] Reason: Other hard stop condition")
                    logger.log_event("WARNING", "terminal_condition", "Other hard stop condition triggered")
                    if discord:
                        discord.notify_risk_warning("critical",
                            "⚠️ HARD STOP CONDITION",
                            "Trading halted")

                print(f"[TERMINAL] All positions should be closed by portfolio controller")
                print(f"[TERMINAL] Stopping trading loop")

                terminal_condition_hit = True
                # Continue to next iteration to verify closure, then exit

            # Step 8: Check if it's time for daily summary
            now = datetime.now()
            if discord and last_summary_date != today:
                # Check if we've passed the summary time
                if now.hour > summary_hour or (now.hour == summary_hour and now.minute >= summary_minute):
                    daily_pnl = account_state.balance - starting_balance
                    daily_pnl_pct = (daily_pnl / starting_balance * 100) if starting_balance > 0 else 0

                    print(f"\n[DAILY SUMMARY] Sending daily summary to Discord...")
                    discord.notify_daily_summary(
                        balance=account_state.balance,
                        equity=account_state.equity,
                        daily_pnl=daily_pnl,
                        daily_pnl_pct=daily_pnl_pct,
                        trades_today=trades_today,
                        open_positions=len(account_state.open_positions),
                    )
                    last_summary_date = today
                    logger.log_event("INFO", "daily_summary", f"Daily summary sent: P&L=${daily_pnl:.2f} ({daily_pnl_pct:.2f}%), Trades={trades_today}")

            # Step 9: Sleep until next 15-minute candle close + buffer
            now = datetime.now()
            # Calculate minutes until next 15-min mark (00, 15, 30, 45)
            minutes_past = now.minute % 15
            if minutes_past == 0 and now.second < 10:
                # We just ran, wait for next 15-min mark
                minutes_to_wait = 15
            else:
                minutes_to_wait = 15 - minutes_past

            # Calculate exact seconds to wait: next 15-min mark + 5 seconds buffer
            seconds_to_wait = (minutes_to_wait * 60) - now.second + 5
            if seconds_to_wait < 10:
                seconds_to_wait += 900  # Add 15 min if we're too close

            next_check_min = ((now.minute // 15) + 1) * 15
            if next_check_min >= 60:
                next_check_str = f"{(now.hour + 1) % 24:02d}:00:05"
            else:
                next_check_str = f"{now.hour:02d}:{next_check_min:02d}:05"

            print(f"\n[SLEEP] Next check at {next_check_str} (in {seconds_to_wait}s)")
            time.sleep(seconds_to_wait)

    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPT] Caught Ctrl+C, shutting down gracefully...")
        logger.log_event("INFO", "bot_shutdown", "Bot stopped by user (Ctrl+C)")
        if discord:
            discord.notify_bot_status("shutdown", f"Bot stopped by user after {iteration} iterations")
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error in main loop: {e}")
        import traceback
        error_details = traceback.format_exc()
        traceback.print_exc()
        logger.log_event("ERROR", "bot_crash", f"Unexpected error: {e}", details=error_details)
        if discord:
            discord.notify_error("bot_crash", f"Bot crashed: {e}", details=error_details[:500])
    finally:
        # Cleanup
        print(f"\n{'=' * 80}")
        print("SHUTDOWN")
        print(f"{'=' * 80}")
        print(f"Total iterations: {iteration}")
        logger.log_event("INFO", "bot_shutdown", f"Bot shutting down after {iteration} iterations")

        # Send shutdown notification if not already sent
        if discord and iteration > 0:
            discord.notify_bot_status("shutdown", f"Bot shutting down after {iteration} iterations")

        # Final account state
        try:
            final_state = mt5_bridge.fetch_account_state(
                portfolio_cfg, risk_cfg, ftmo_cfg
            )
            print(f"\nFinal Account State:")
            print(f"  Equity:  ${final_state.equity:,.2f}")
            print(f"  Balance: ${final_state.balance:,.2f}")
            print(f"  Open positions: {len(final_state.open_positions)}")
        except Exception as e:
            print(f"[WARNING] Could not fetch final state: {e}")

        mt5_bridge.shutdown_mt5()
        print(f"\n[DONE] Shutdown complete")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Live MT5 paper trading runner for FTMO eval mode",
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
        "--poll-interval-seconds",
        type=int,
        default=60,
        help="Loop frequency in seconds (default: 60)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Max loop iterations (default: 0 = unlimited, useful for testing)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: generate signals and compute orders but do NOT send to MT5"
    )

    args = parser.parse_args()

    # Validate poll interval
    if args.poll_interval_seconds < 1:
        print(f"[ERROR] Poll interval must be >= 1 second")
        sys.exit(1)

    # Run live loop
    run_live_loop(
        mode=args.mode,
        poll_interval_seconds=args.poll_interval_seconds,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
