# risk_manager.py
"""
Risk management module for portfolio trading.

Handles:
- Daily drawdown calculation
- Open risk tracking
- FTMO limit checks
- Risk envelope guard
- Eval-phase profit target lock
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from config_loader import FTMOOverlayConfig


# ---------------------------------------------------------------------------
# Core state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Position:
    instrument: str
    direction: str          # "long" or "short"
    entry_price: float
    sl_price: float
    size_lots: float
    risk_pct: float         # risk as % of equity when opened (1R = this)


@dataclass
class AccountState:
    equity: float
    balance: float
    start_of_day_equity: float
    realised_pnl_today: float
    open_positions: List[Position]
    floating_pnl: float = 0.0  # Unrealized P&L (equity - balance), negative = loss


# ---------------------------------------------------------------------------
# Helper calculations
# ---------------------------------------------------------------------------

def compute_daily_dd_pct(state: AccountState) -> float:
    """
    Daily drawdown as a percentage of start-of-day equity.

    Positive number means a loss from start_of_day_equity.
    """
    sod = state.start_of_day_equity
    if sod <= 0:
        return 0.0
    dd = (sod - state.equity) / sod * 100.0
    return max(dd, 0.0)


def compute_open_risk_pct(state: AccountState) -> float:
    """
    Approximate open risk as sum of per-position risk_pct.

    Assumes each position's risk_pct is 1R for that trade.
    """
    return float(sum(pos.risk_pct for pos in state.open_positions))


# ---------------------------------------------------------------------------
# Main FTMO risk evaluation
# ---------------------------------------------------------------------------

def evaluate_daily_risk_state(
    state: AccountState,
    ftmo: FTMOOverlayConfig,
    mode: str = "eval",
) -> Dict[str, Any]:
    """
    Evaluate current account state against FTMO-style limits.

    Returns a dict with:
        dd_today_pct
        open_risk_pct
        combined_risk_pct

        breached_daily_limit
        breached_total_limit

        envelope_guard_triggered
        soft_brake_triggered
        hard_brake_triggered

        hit_eval_profit_target

        block_new_trades
        must_flatten
    """
    limits = ftmo.limits

    # --- core measures ---
    dd_today_pct = compute_daily_dd_pct(state)
    open_risk_pct = compute_open_risk_pct(state)
    combined_risk_pct = dd_today_pct + open_risk_pct

    # --- daily limit (from start-of-day) ---
    daily_limit = limits.internal_daily_loss_limit_pct
    breached_daily_limit = dd_today_pct >= daily_limit

    # --- total limit (from eval starting equity or some baseline) ---
    if mode == "eval" and getattr(ftmo, "eval_phase", None) is not None:
        ref_equity = ftmo.eval_phase.starting_equity
    else:
        ref_equity = state.start_of_day_equity

    if ref_equity <= 0:
        total_dd_pct = 0.0
    else:
        total_dd_pct = max((ref_equity - state.equity) / ref_equity * 100.0, 0.0)

    total_limit = limits.internal_total_loss_limit_pct
    breached_total_limit = total_dd_pct >= total_limit

    # --- intraday soft/hard brakes on realised DD only ---
    # Derive from daily_limit:
    #   soft ≈ 50% of daily_limit
    #   hard ≈ 90% of daily_limit
    soft_threshold = daily_limit * 0.5
    hard_threshold = daily_limit * 0.9

    soft_brake_triggered = dd_today_pct >= soft_threshold
    hard_brake_triggered = dd_today_pct >= hard_threshold

    # --- Risk Envelope Guard: realised DD + open risk must never exceed limit - margin ---
    env = getattr(ftmo, "risk_envelope_guard", None)
    if env is not None and env.enabled:
        margin = env.safety_margin_pct
        guard_threshold = daily_limit - margin
        envelope_guard_triggered = combined_risk_pct >= guard_threshold
    else:
        envelope_guard_triggered = False

    # --- Eval-phase profit target lock ---
    hit_eval_profit_target = False
    eval_cfg = getattr(ftmo, "eval_phase", None)
    if mode == "eval" and eval_cfg is not None and eval_cfg.enabled:
        target_equity = eval_cfg.starting_equity * (1.0 + eval_cfg.profit_target_pct / 100.0)
        # Use tolerance for floating point comparison (within $0.01)
        if eval_cfg.lock_trading_on_target and state.balance >= (target_equity - 0.01):
            hit_eval_profit_target = True

    # --- Unrealized loss limit (Blueguardian specific, 0 = disabled for FTMO) ---
    max_unrealised_limit = getattr(limits, "internal_max_unrealised_loss_pct", 0.0)
    if max_unrealised_limit > 0:
        # Only check if we have floating loss (negative floating_pnl)
        if state.floating_pnl < 0 and state.balance > 0:
            # Calculate floating loss as % of current balance (strictest basis)
            unrealised_loss_pct = abs(state.floating_pnl) / state.balance * 100.0
        else:
            unrealised_loss_pct = 0.0
        breached_unrealised_limit = unrealised_loss_pct >= max_unrealised_limit
    else:
        unrealised_loss_pct = 0.0
        breached_unrealised_limit = False

    # -------------------------------------------------------------------
    # Decide actions: block_new_trades / must_flatten
    # -------------------------------------------------------------------
    block_new_trades = False
    must_flatten = False

    # Hard brakes: flatten + no new trades
    hard_conditions = any([
        breached_total_limit,
        breached_daily_limit,
        hard_brake_triggered,
        envelope_guard_triggered,      # envelope guard must ALWAYS flatten
        hit_eval_profit_target,        # eval profit lock
        breached_unrealised_limit,     # Blueguardian unrealized loss limit
    ])

    if hard_conditions:
        must_flatten = True
        block_new_trades = True
    else:
        if soft_brake_triggered:
            block_new_trades = True

    risk_state: Dict[str, Any] = {
        "dd_today_pct": dd_today_pct,
        "open_risk_pct": open_risk_pct,
        "combined_risk_pct": combined_risk_pct,
        "total_dd_pct": total_dd_pct,
        "unrealised_loss_pct": unrealised_loss_pct,

        "breached_daily_limit": breached_daily_limit,
        "breached_total_limit": breached_total_limit,
        "soft_brake_triggered": soft_brake_triggered,
        "hard_brake_triggered": hard_brake_triggered,
        "envelope_guard_triggered": envelope_guard_triggered,
        "hit_eval_profit_target": hit_eval_profit_target,
        "breached_unrealised_limit": breached_unrealised_limit,

        "block_new_trades": block_new_trades,
        "must_flatten": must_flatten,
    }

    return risk_state


# ---------------------------------------------------------------------------
# Pretty-printers for logs / debugging
# ---------------------------------------------------------------------------

def describe_risk_state(risk_state: Dict[str, Any]) -> str:
    """
    Turn a risk_state dict into a human-readable summary.
    """
    lines = []
    lines.append("Risk State:")
    lines.append(f"  DD today:     {risk_state['dd_today_pct']:.2f}%")
    lines.append(f"  Open risk:    {risk_state['open_risk_pct']:.2f}%")
    lines.append(f"  Combined:     {risk_state['combined_risk_pct']:.2f}%")
    lines.append(f"  Total DD:     {risk_state['total_dd_pct']:.2f}%")
    # Only show unrealized loss if it's being tracked (> 0)
    unrealised = risk_state.get('unrealised_loss_pct', 0.0)
    if unrealised > 0:
        lines.append(f"  Unrealised:   {unrealised:.2f}%")
    lines.append(f"  Eval target hit: {risk_state.get('hit_eval_profit_target', False)}")

    if risk_state["breached_total_limit"]:
        lines.append("  Status: BREACHED_TOTAL_LIMIT")
    elif risk_state["breached_daily_limit"]:
        lines.append("  Status: BREACHED_DAILY_LIMIT")
    elif risk_state.get("breached_unrealised_limit", False):
        lines.append("  Status: BREACHED_UNREALISED_LIMIT")
    elif risk_state["hard_brake_triggered"]:
        lines.append("  Status: HARD_BRAKE_TRIGGERED")
    elif risk_state["envelope_guard_triggered"]:
        lines.append("  Status: ENVELOPE_GUARD_TRIGGERED")
    elif risk_state["hit_eval_profit_target"]:
        lines.append("  Status: EVAL_PROFIT_TARGET_HIT")
    elif risk_state["soft_brake_triggered"]:
        lines.append("  Status: SOFT_BRAKE_TRIGGERED")
    else:
        lines.append("  Status: OK")

    if risk_state["must_flatten"]:
        lines.append("  Action: MUST_FLATTEN")
    elif risk_state["block_new_trades"]:
        lines.append("  Action: BLOCK_NEW_TRADES")
    else:
        lines.append("  Action: ALLOW_TRADING")

    return "\n".join(lines)
