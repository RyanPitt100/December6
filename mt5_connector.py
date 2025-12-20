# mt5_connector.py

from typing import List, Optional
from datetime import datetime

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    print("[mt5_connector] MetaTrader5 package not installed; live trading disabled.")


def mt5_initialize(login: int = None, password: str = None, server: str = None) -> bool:
    if mt5 is None:
        return False

    if not mt5.initialize():
        print(f"[mt5] initialize() failed, error code = {mt5.last_error()}")
        return False

    if login and password and server:
        authorized = mt5.login(login=login, password=password, server=server)
        if not authorized:
            print(f"[mt5] login() failed, error code = {mt5.last_error()}")
            return False

    print("[mt5] Connected.")
    return True


def mt5_shutdown():
    if mt5 is not None:
        mt5.shutdown()


def get_ohlc_from_mt5(
    symbol: str, timeframe: int, n_bars: int = 300
) -> Optional[pd.DataFrame]:
    """
    Pull recent OHLCV data from MT5 for the given symbol/timeframe.
    Timeframe should be mt5.TIMEFRAME_M15, etc.
    """

    if mt5 is None:
        print("[mt5] Not available.")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        print(f"[mt5] copy_rates_from_pos failed for {symbol}, err={mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize("UTC")
    df.set_index("time", inplace=True)
    df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "tick_volume",
        },
        inplace=True,
    )
    return df


def place_market_order(
    symbol: str,
    direction: str,
    volume: float,
    sl: float,
    tp: float,
    comment: str = "",
):
    """
    Place a market order via MT5.
    direction: "LONG" or "SHORT"
    """

    if mt5 is None:
        print("[mt5] Not available, dry-run only.")
        print(
            f"[DRY-RUN ORDER] {direction} {symbol} vol={volume}, sl={sl}, tp={tp}, comment={comment}"
        )
        return

    if direction.upper() == "LONG":
        order_type = mt5.ORDER_TYPE_BUY
    else:
        order_type = mt5.ORDER_TYPE_SELL

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[mt5] No tick for {symbol}, cannot place order.")
        return

    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result is None:
        print(f"[mt5] order_send failed for {symbol}, err={mt5.last_error()}")
        return

    print(f"[mt5] order_send result: {result}")
