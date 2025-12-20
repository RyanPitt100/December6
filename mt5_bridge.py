"""
mt5_bridge.py

Minimal MetaTrader 5 bridge for live trading.

This module provides a clean interface to MT5 without implementing any trading logic.
All risk management, position sizing, and FTMO overlays are handled by the existing
risk_manager.py and portfolio_controller.py modules.

Requirements:
    - MetaTrader5 Python package: pip install MetaTrader5
    - MT5 terminal running and logged in
    - Credentials in config/credentials.yml
"""

import MetaTrader5 as mt5
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time
import yaml
from pathlib import Path
import pandas as pd

from risk_manager import AccountState, Position
from portfolio_controller import Order


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def load_credentials() -> Dict[str, any]:
    """
    Load MT5 credentials from config/credentials.yml.

    Expected structure:
        mt5:
          login: 1520859470
          password: "SECRET"
          server: "FTMO-Demo2"

    Returns:
        Dictionary with 'login', 'password', 'server' keys

    Raises:
        FileNotFoundError: If credentials.yml not found
        KeyError: If mt5 section missing
    """
    creds_path = Path("config/credentials.yml")
    if not creds_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {creds_path}")

    with open(creds_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'mt5' not in config:
        raise KeyError("'mt5' section not found in credentials.yml")

    mt5_config = config['mt5']
    required_keys = ['login', 'password', 'server']
    for key in required_keys:
        if key not in mt5_config:
            raise KeyError(f"Missing required MT5 credential: {key}")

    return mt5_config


def init_mt5() -> None:
    """
    Initialize connection to MetaTrader 5 using credentials from config/credentials.yml.

    Raises:
        RuntimeError: If MT5 initialization or login fails
    """
    # Load credentials
    creds = load_credentials()
    login = int(creds['login'])
    password = str(creds['password'])
    server = str(creds['server'])

    print(f"[MT5] Initializing connection...")
    print(f"[MT5] Login: {login}")
    print(f"[MT5] Server: {server}")

    # Initialize MT5
    if not mt5.initialize():
        error = mt5.last_error()
        raise RuntimeError(f"MT5 initialization failed: {error}")

    print(f"[MT5] MT5 initialized successfully")

    # Login
    authorized = mt5.login(login=login, password=password, server=server)
    if not authorized:
        error = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"MT5 login failed: {error}")

    print(f"[MT5] Login successful")

    # Print account info
    account_info = mt5.account_info()
    if account_info is not None:
        print(f"[MT5] Account: {account_info.name}")
        print(f"[MT5] Balance: ${account_info.balance:,.2f}")
        print(f"[MT5] Equity: ${account_info.equity:,.2f}")
        print(f"[MT5] Leverage: 1:{account_info.leverage}")
    else:
        print(f"[MT5] Warning: Could not fetch account info")


def shutdown_mt5() -> None:
    """Cleanly shutdown MT5 connection."""
    mt5.shutdown()
    print(f"[MT5] Connection closed")


# ---------------------------------------------------------------------------
# Account State
# ---------------------------------------------------------------------------

def fetch_account_state(portfolio_cfg, risk_cfg, ftmo_cfg) -> AccountState:
    """
    Fetch current account state from MT5 and populate AccountState dataclass.

    This function bridges MT5 data to the existing risk_manager.AccountState format.

    Args:
        portfolio_cfg: Portfolio configuration (for instrument mapping)
        risk_cfg: Risk configuration (unused here, passed for consistency)
        ftmo_cfg: FTMO configuration (unused here, passed for consistency)

    Returns:
        AccountState object with current MT5 account data

    Implementation notes:
        - equity: Direct from MT5 account_info.equity
        - balance: Direct from MT5 account_info.balance
        - start_of_day_equity: Approximated as balance (TODO: refine with daily history)
        - realised_pnl_today: Approximated as (balance - start_of_day_balance)
          * For now, we use balance as proxy for start_of_day
          * TODO: Fetch actual daily trade history and sum closed PnL
        - open_positions: Converted from MT5 positions to Position objects
          * risk_pct: Approximated from (SL distance * volume * pip_value / equity)
          * This is an estimate; actual risk_pct should ideally come from order history
    """
    # Fetch account info
    account_info = mt5.account_info()
    if account_info is None:
        raise RuntimeError(f"Failed to fetch MT5 account info: {mt5.last_error()}")

    equity = account_info.equity
    balance = account_info.balance

    # Approximate start_of_day_equity
    # TODO: Improve by fetching balance at start of day from daily history
    start_of_day_equity = balance

    # Approximate realised_pnl_today
    # TODO: Improve by summing closed trade PnL for today
    # For now, use balance - start_of_day as proxy
    realised_pnl_today = balance - start_of_day_equity

    # Fetch open positions
    positions = mt5.positions_get()
    if positions is None:
        positions = []  # No positions

    open_positions = []
    for pos in positions:
        # Map MT5 position to our Position dataclass
        direction = "long" if pos.type == mt5.ORDER_TYPE_BUY else "short"

        # Approximate risk_pct from SL distance
        # risk_pct = (SL distance in pips * pip_value * volume) / equity * 100
        # This is a rough estimate; ideally we'd track this from order history
        sl_distance = abs(pos.price_open - pos.sl) if pos.sl > 0 else 0.0

        # For forex pairs, approximate pip value (this is simplified)
        # TODO: Use proper contract specifications for accurate pip value
        if sl_distance > 0:
            # Rough approximation: assume $10 per pip per standard lot for majors
            pip_value_per_lot = 10.0  # USD per pip per standard lot
            pips = sl_distance / 0.0001  # Assume 4-decimal pairs
            risk_dollars = pips * pip_value_per_lot * pos.volume
            risk_pct = (risk_dollars / equity) * 100.0
        else:
            risk_pct = 0.5  # Default fallback (will be refined)

        position = Position(
            instrument=pos.symbol,
            direction=direction,
            entry_price=pos.price_open,
            sl_price=pos.sl if pos.sl > 0 else pos.price_open * 0.98,  # Fallback SL
            size_lots=pos.volume,
            risk_pct=risk_pct,
        )
        open_positions.append(position)

    # Calculate floating P&L (unrealized profit/loss)
    floating_pnl = equity - balance  # Positive = profit, negative = loss

    return AccountState(
        equity=equity,
        balance=balance,
        start_of_day_equity=start_of_day_equity,
        realised_pnl_today=realised_pnl_today,
        open_positions=open_positions,
        floating_pnl=floating_pnl,
    )


# ---------------------------------------------------------------------------
# Price Data
# ---------------------------------------------------------------------------

def fetch_prices(symbols: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Fetch current bid/ask prices for given symbols.

    Args:
        symbols: List of MT5 symbol names (e.g., ["EURUSD", "GBPUSD"])

    Returns:
        Dictionary mapping symbol -> (bid, ask) tuple

    Note:
        Returns empty dict if no ticks available (e.g., market closed)
    """
    prices = {}

    for symbol in symbols:
        # Get current tick
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"[MT5] Warning: No tick data for {symbol}")
            continue

        prices[symbol] = (tick.bid, tick.ask)

    return prices


def fetch_mid_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch current mid prices for given symbols.

    Args:
        symbols: List of MT5 symbol names

    Returns:
        Dictionary mapping symbol -> mid_price
    """
    prices = fetch_prices(symbols)
    return {symbol: (bid + ask) / 2.0 for symbol, (bid, ask) in prices.items()}


# ---------------------------------------------------------------------------
# OHLC Data Fetching
# ---------------------------------------------------------------------------

def fetch_ohlc(
    symbol: str,
    timeframe: str,
    n_bars: int = 300,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLC data from MT5 for a specific symbol and timeframe.

    Args:
        symbol: MT5 symbol name (e.g., "EURUSD", "JP225.cash")
        timeframe: Timeframe string: "15m", "1h", "4h", "1d"
        n_bars: Number of bars to fetch (default: 300)

    Returns:
        DataFrame with columns: timestamp (index), open, high, low, close, volume
        Returns None if data fetch fails

    Timeframe mapping:
        "15m" -> MT5_TIMEFRAME_M15
        "1h"  -> MT5_TIMEFRAME_H1
        "4h"  -> MT5_TIMEFRAME_H4
        "1d"  -> MT5_TIMEFRAME_D1
    """
    # Map timeframe strings to MT5 constants
    timeframe_map = {
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }

    if timeframe not in timeframe_map:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(timeframe_map.keys())}")

    mt5_timeframe = timeframe_map[timeframe]

    # Fetch bars
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, n_bars)

    if rates is None or len(rates) == 0:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert time column to datetime
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Set timestamp as index
    df = df.set_index('timestamp')

    # Keep only OHLC columns
    df = df[['open', 'high', 'low', 'close', 'tick_volume']].copy()
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)

    return df


def fetch_multi_tf_data(
    symbol: str,
    n_bars_15m: int = 500,
) -> Optional[pd.DataFrame]:
    """
    Fetch multi-timeframe data for a symbol matching the structure from build_multi_tf_frame().

    This fetches 15m OHLC data and higher timeframe data (1h, 4h, 1d) but does NOT
    include regime labels (those require offline labeling). The returned DataFrame
    can be used for indicator calculation but NOT for regime-gated strategies without
    additional regime labeling.

    Args:
        symbol: MT5 symbol name
        n_bars_15m: Number of 15m bars to fetch (default: 500)

    Returns:
        DataFrame with 15m timestamp index and columns:
        - open, high, low, close (15m OHLC)
        - volume (15m volume)

    Note:
        This function does NOT include regime_h1, regime_h4, regime_d1 columns.
        For regime-based strategies, you need to either:
        1. Use precomputed regime labels from labelled CSVs, or
        2. Implement real-time regime detection (future enhancement)

    Usage:
        df = fetch_multi_tf_data("EURUSD", n_bars_15m=500)
        # df will have 15m OHLC data only (no regimes)
        # For now, this is sufficient for testing connectivity
        # Real signal generation will need regime labels
    """
    # Fetch 15m base data
    df_15m = fetch_ohlc(symbol, "15m", n_bars=n_bars_15m)

    if df_15m is None:
        return None

    # TODO: Add regime labeling or merge with precomputed regime data
    # For now, return just OHLC (caller must handle missing regime columns)

    return df_15m


# ---------------------------------------------------------------------------
# Order Execution
# ---------------------------------------------------------------------------

def send_orders(orders: List[Order]) -> None:
    """
    Execute orders on MT5.

    This function maps Order objects from portfolio_controller to MT5 market orders.

    Args:
        orders: List of Order objects to execute

    Order types:
        - action="open": Open new position at market
          * Uses order.direction ("long" or "short")
          * Attaches SL/TP if provided
          * Uses order.size_lots for volume
        - action="close": Close existing position
          * Closes position for order.instrument
          * Uses order.size_lots if partial close needed

    Note:
        All risk sizing is already done by portfolio_controller.
        This function only executes what it's told to do.
    """
    if not orders:
        return

    print(f"\n[MT5] Executing {len(orders)} order(s)...")

    for order in orders:
        try:
            if order.action == "open":
                _execute_open_order(order)
            elif order.action == "close":
                _execute_close_order(order)
            else:
                print(f"[MT5] Warning: Unknown order action '{order.action}'")

        except Exception as e:
            print(f"[MT5] ERROR executing order for {order.instrument}: {e}")
            print(f"[MT5]   Order details: {order}")


def _execute_open_order(order: Order) -> None:
    """
    Open a new position at market.

    Args:
        order: Order object with action="open"
    """
    symbol = order.instrument
    volume = order.size_lots
    direction = order.direction

    # Ensure symbol is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError(f"Symbol {symbol} not found in MT5")

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol {symbol}")

    # Determine order type
    if direction == "long":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif direction == "short":
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # Prepare request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": 20,  # Max slippage in points
        "magic": 234000,  # Magic number for our bot
        "comment": f"bot_{order.reason}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Add SL/TP if provided
    if order.sl_price is not None and order.sl_price > 0:
        request["sl"] = order.sl_price
    if order.tp_price is not None and order.tp_price > 0:
        request["tp"] = order.tp_price

    # Send order
    result = mt5.order_send(request)

    if result is None:
        raise RuntimeError(f"order_send failed: {mt5.last_error()}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(
            f"Order failed with retcode {result.retcode}: {result.comment}"
        )

    print(f"[MT5] OPEN {direction.upper()} {symbol} {volume:.2f} lots @ {price:.5f}")
    if order.sl_price:
        print(f"[MT5]   SL: {order.sl_price:.5f}")
    if order.tp_price:
        print(f"[MT5]   TP: {order.tp_price:.5f}")


def _execute_close_order(order: Order) -> None:
    """
    Close an existing position.

    Args:
        order: Order object with action="close"
    """
    symbol = order.instrument

    # Get open positions for this symbol
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        print(f"[MT5] Warning: No open position for {symbol} to close")
        return

    # Close each position (usually just one per symbol)
    for position in positions:
        # Determine close order type (opposite of position type)
        if position.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"bot_close_{order.reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send close order
        result = mt5.order_send(request)

        if result is None:
            raise RuntimeError(f"order_send failed: {mt5.last_error()}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(
                f"Close order failed with retcode {result.retcode}: {result.comment}"
            )

        print(f"[MT5] CLOSE {symbol} {position.volume:.2f} lots @ {price:.5f} (reason: {order.reason})")


# ---------------------------------------------------------------------------
# Market Context Helpers
# ---------------------------------------------------------------------------

def get_spread_points(symbol: str) -> Optional[float]:
    """
    Get current spread in points for a symbol.

    Args:
        symbol: MT5 symbol name

    Returns:
        Spread in points, or None if unavailable
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None

    # Spread in points = (ask - bid) / point
    spread_points = (tick.ask - tick.bid) / symbol_info.point
    return spread_points


def get_symbol_info(symbol: str) -> Optional[Dict]:
    """
    Get symbol information including point value, digits, etc.

    Args:
        symbol: MT5 symbol name

    Returns:
        Dict with symbol info, or None if unavailable
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return None

    return {
        "symbol": info.name,
        "point": info.point,
        "digits": info.digits,
        "spread": info.spread,
        "trade_contract_size": info.trade_contract_size,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
    }
