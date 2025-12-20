# portfolio_controller.py
"""
Portfolio controller for trade decision making.

Integrates:
- Trade signals
- Risk management
- Position sizing
- FTMO overlays

Produces final order list.
"""

from dataclasses import dataclass
from typing import List, Optional

from config_loader import (
    InstrumentPortfolioConfig,
    PortfolioRiskConfig,
    FTMOOverlayConfig,
)
from risk_manager import (
    AccountState,
    Position,
    evaluate_daily_risk_state,
    compute_open_risk_pct,
)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    instrument: str
    direction: str          # "long" or "short"
    entry_price: float
    sl_price: float
    tp_price: Optional[float] = None
    reason: str = "signal"


@dataclass
class Order:
    instrument: str
    action: str             # "open", "close", "reduce"
    direction: Optional[str]
    size_lots: float
    entry_price: Optional[float]
    sl_price: Optional[float]
    tp_price: Optional[float]
    reason: str


# ---------------------------------------------------------------------------
# Position sizing helper (very simplified placeholder)
# ---------------------------------------------------------------------------

def _calc_position_size_lots(
    equity: float,
    risk_pct: float,
    entry_price: float,
    sl_price: float,
    instrument: str = "",
) -> float:
    """
    Calculate position size in lots based on risk percentage.

    For forex pairs (XXXYYY):
    - 1 standard lot = 100,000 units
    - 1 pip = 0.0001 for 4-decimal pairs, 0.01 for 2-decimal pairs (JPY)
    - Pip value = (0.0001 / quote_price) * 100,000 for quote in USD
    - For USD quote pairs: pip value ≈ $10 per lot

    For indices (e.g., JP225.cash):
    - 1 lot = varies by broker/contract
    - Point value = varies by contract

    Args:
        equity: Account equity in USD
        risk_pct: Risk per trade as percentage (e.g., 0.5 for 0.5%)
        entry_price: Entry price
        sl_price: Stop loss price
        instrument: Instrument symbol (e.g., "EURUSD", "JP225.cash")

    Returns:
        Position size in lots
    """
    risk_value = equity * (risk_pct / 100.0)
    price_diff = abs(entry_price - sl_price)

    if price_diff <= 0:
        return 0.0

    # Determine pip/point value based on instrument type
    if "JPY" in instrument.upper():
        # JPY pairs: 2-decimal, pip = 0.01
        pip_size = 0.01
        pip_value_per_lot = 10.0  # Approximate for XXXJPY pairs
    elif any(idx in instrument.upper() for idx in ["JP225", "AUS200", "GER40", "US30", "NAS100"]):
        # Indices: point value varies
        # For most CFD indices on FTMO: 1 lot = $1 per point typically
        pip_size = 1.0
        pip_value_per_lot = 1.0
    else:
        # Standard forex (4-decimal pairs): pip = 0.0001
        pip_size = 0.0001
        pip_value_per_lot = 10.0  # Standard for XXXUSD pairs

    # Calculate SL distance in pips/points
    sl_distance_pips = price_diff / pip_size

    # Position size = risk_value / (sl_distance_pips * pip_value_per_lot)
    if sl_distance_pips <= 0:
        return 0.0

    size_lots = risk_value / (sl_distance_pips * pip_value_per_lot)

    # Round to broker's acceptable lot size increment
    # Most brokers: 0.01 lots minimum increment
    # Some micro brokers: 0.001 lots
    # FTMO typically uses 0.01 lot increments
    size_lots = round(size_lots, 2)

    # Ensure minimum position size (0.01 lots for most brokers)
    if size_lots < 0.01:
        size_lots = 0.01

    return size_lots


# ---------------------------------------------------------------------------
# Main portfolio decision function
# ---------------------------------------------------------------------------

def decide_portfolio_orders(
    account_state: AccountState,
    signals: List[TradeSignal],
    portfolio_cfg: InstrumentPortfolioConfig,
    risk_cfg: PortfolioRiskConfig,
    ftmo_cfg: FTMOOverlayConfig,
    mode: str = "eval",  # "eval" or "funded"
) -> List[Order]:
    """
    Turn signals + risk state into final orders.

    Behaviour:
    - If any hard risk condition is hit (daily / total / envelope / eval-lock):
        -> Flatten all positions, no new trades.
    - Else:
        -> Honour soft brakes (block_new_trades).
        -> Respect max_open_trades and max_risk_at_once_pct.
        -> Size each trade using risk_per_trade_pct from risk_cfg.
    """
    orders: List[Order] = []

    # 1) Evaluate risk state
    risk_state = evaluate_daily_risk_state(account_state, ftmo_cfg, mode=mode)

    # Helper: flatten all positions
    def _flatten_all(reason: str) -> None:
        for pos in account_state.open_positions:
            orders.append(
                Order(
                    instrument=pos.instrument,
                    action="close",
                    direction=None,
                    size_lots=pos.size_lots,
                    entry_price=None,
                    sl_price=None,
                    tp_price=None,
                    reason=reason,
                )
            )

    # Hard brakes: flatten + no new trades
    if (
        risk_state["breached_total_limit"]
        or risk_state["breached_daily_limit"]
        or risk_state["hard_brake_triggered"]
        or risk_state["envelope_guard_triggered"]
        or risk_state["hit_eval_profit_target"]
    ):
        _flatten_all("risk_hard_brake_or_eval_target")
        return orders

    # Soft brake: don't open new trades, but we don't auto-flatten
    if risk_state["soft_brake_triggered"]:
        # No new trades, just maintain / let other modules manage
        return orders

    # 2) Normal trading: open new positions within risk constraints
    # -------------------------------------------------------------
    # Risk per trade (both eval and funded use 0.5% if configured that way)
    if mode == "eval":
        risk_per_trade_pct = risk_cfg.risk_per_trade_eval_pct
    else:
        risk_per_trade_pct = risk_cfg.risk_per_trade_funded_pct

    # Current open risk and limits
    current_open_risk = compute_open_risk_pct(account_state)
    max_risk_at_once = risk_cfg.max_risk_at_once_pct
    max_open_trades = risk_cfg.max_open_trades

    # Set of tradable instruments from portfolio config
    tradable = {inst.symbol for inst in portfolio_cfg.instruments}

    open_trade_count = len(account_state.open_positions)

    for sig in signals:
        if sig.instrument not in tradable:
            continue

        # Respect max open trades
        if open_trade_count >= max_open_trades:
            break

        # Check risk-at-once budget
        projected_open_risk = current_open_risk + risk_per_trade_pct
        if projected_open_risk > max_risk_at_once:
            # Skip this trade, not enough risk budget
            continue

        # Position sizing
        size_lots = _calc_position_size_lots(
            equity=account_state.equity,
            risk_pct=risk_per_trade_pct,
            entry_price=sig.entry_price,
            sl_price=sig.sl_price,
            instrument=sig.instrument,
        )
        if size_lots <= 0:
            continue

        orders.append(
            Order(
                instrument=sig.instrument,
                action="open",
                direction=sig.direction,
                size_lots=size_lots,
                entry_price=sig.entry_price,
                sl_price=sig.sl_price,
                tp_price=sig.tp_price,
                reason=sig.reason,
            )
        )

        # Update counters
        current_open_risk += risk_per_trade_pct
        open_trade_count += 1

    return orders


# ---------------------------------------------------------------------------
# Pretty-printer for orders
# ---------------------------------------------------------------------------

def describe_orders(orders: List[Order]) -> str:
    """
    Render a list of orders into a human-readable string, useful for logging.
    """
    if not orders:
        return "Orders: NONE"

    lines = ["Orders:"]
    for order in orders:
        if order.action == "open":
            lines.append(
                f"  [OPEN] {order.instrument} {order.direction} {order.size_lots:.2f} lots "
                f"@ {order.entry_price:.5f} (SL: {order.sl_price:.5f})"
            )
        elif order.action == "close":
            lines.append(
                f"  [CLOSE] {order.instrument} {order.size_lots:.2f} lots (reason: {order.reason})"
            )
        elif order.action == "reduce":
            lines.append(
                f"  [REDUCE] {order.instrument} by {order.size_lots:.2f} lots (reason: {order.reason})"
            )
        else:
            lines.append(
                f"  [{order.action.upper()}] {order.instrument} {order.size_lots:.2f} lots"
            )

    return "\n".join(lines)
