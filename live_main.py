# live_main.py

import yaml
from datetime import datetime, timezone

from live_engine import generate_latest_trend_signal
from execution_risk import AccountState, PropRules, can_open_new_trade
from mt5_connector import mt5_initialize, mt5_shutdown, place_market_order
from strategies import Signal


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml("live_config.yml")
    universe = load_yaml("universe.yml")

    data_root = cfg["data"]["data_root"]

    # For now, mock account state; later we'll pull these from MT5.
    account = AccountState(
        equity=cfg["account"]["initial_equity"],
        balance=cfg["account"]["initial_equity"],
        start_of_day_equity=cfg["account"]["initial_equity"],
        peak_equity=cfg["account"]["initial_equity"],
        open_risk_pct=0.0,
        todays_pnl_pct=0.0,
    )

    rules = PropRules(
        daily_loss_limit_pct=cfg["ftmo_rules"]["daily_loss_limit_pct"],
        max_relative_dd_pct=cfg["ftmo_rules"]["max_relative_dd_pct"],
        max_total_risk_pct=cfg["risk"]["max_total_risk_pct"],
        safety_buffer_pct=cfg["ftmo_rules"]["safety_buffer_pct"],
        weekend_flatten=cfg["ftmo_rules"]["weekend_flatten"],
    )

    # In real live mode you would uncomment and fill login details.
    # mt5_initialize(login=..., password=..., server=...)

    now = datetime.now(timezone.utc)

    for inst_cfg in universe["instruments"]:
        symbol = inst_cfg["symbol"]
        if not inst_cfg.get("enabled", True):
            continue

        if "Trend_EMA_Pullback" not in inst_cfg.get("strategies", []):
            continue

        print(f"\n[live_main] Checking {symbol}...")

        sig: Signal | None = generate_latest_trend_signal(
            instrument=symbol,
            data_root=data_root,
        )

        if sig is None:
            print(f"[live_main] No signal for {symbol}.")
            continue

        # For now, we treat each trade as cfg["risk"]["per_trade_risk_pct"] of equity.
        per_trade_risk = cfg["risk"]["per_trade_risk_pct"]
        if not can_open_new_trade(account, rules, proposed_risk_pct=per_trade_risk, now=now):
            print(f"[live_main] Risk guard blocked trade on {symbol}.")
            continue

        # Dry-run: just print the order details.
        print(
            f"[live_main] SIGNAL {symbol}: {sig.direction} at {sig.entry_price}, "
            f"SL={sig.stop_loss}, TP={sig.take_profit}, regime={sig.regime}"
        )

        # Later, when you’re happy, you can uncomment this to actually submit orders via MT5:
        # place_market_order(
        #     symbol=symbol,
        #     direction=sig.direction,
        #     volume=calculated_volume,
        #     sl=sig.stop_loss,
        #     tp=sig.take_profit,
        #     comment=sig.strategy_name,
        # )

    # mt5_shutdown()


if __name__ == "__main__":
    main()
