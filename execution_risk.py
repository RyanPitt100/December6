# execution_risk.py

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timezone


@dataclass
class AccountState:
    equity: float
    balance: float
    # Start-of-day equity for FTMO-style daily loss limits.
    start_of_day_equity: float
    # Max peak equity since account start (for total dd).
    peak_equity: float
    # Sum of open risk in % of equity (roughly).
    open_risk_pct: float
    # Today's realised PnL in % of start-of-day equity.
    todays_pnl_pct: float


@dataclass
class PropRules:
    daily_loss_limit_pct: float
    max_relative_dd_pct: float
    max_total_risk_pct: float = 1.0  # max open risk across all positions
    safety_buffer_pct: float = 0.5
    weekend_flatten: bool = True


def is_weekend(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = datetime.now(timezone.utc)
    return now.weekday() >= 5  # 5 = Saturday, 6 = Sunday


def can_open_new_trade(
    account: AccountState,
    rules: PropRules,
    proposed_risk_pct: float,
    now: Optional[datetime] = None,
) -> bool:
    """
    Decides if we are allowed to open a new position under FTMO-like rules.
    """

    if now is None:
        now = datetime.now(timezone.utc)

    # 1) Weekend flatten / block
    if rules.weekend_flatten and is_weekend(now):
        print("[risk] Weekend – blocking new trades.")
        return False

    # 2) Daily loss limit check
    # Today's realised PnL as % of start-of-day equity.
    # Negative PnL reduces our room.
    effective_daily_limit = rules.daily_loss_limit_pct - rules.safety_buffer_pct
    if account.todays_pnl_pct <= -effective_daily_limit:
        print(
            f"[risk] Daily loss limit hit: todays_pnl={account.todays_pnl_pct:.2f}% "
            f"<= -{effective_daily_limit:.2f}%"
        )
        return False

    # 3) Overall max relative drawdown from peak equity
    if account.peak_equity > 0:
        current_dd_pct = (account.equity - account.peak_equity) / account.peak_equity * 100.0
        if current_dd_pct <= -rules.max_relative_dd_pct:
            print(
                f"[risk] Max relative DD hit: dd={current_dd_pct:.2f}% "
                f"<= -{rules.max_relative_dd_pct:.2f}%"
            )
            return False

    # 4) Total open risk cap
    new_total_risk = account.open_risk_pct + proposed_risk_pct
    if new_total_risk > rules.max_total_risk_pct:
        print(
            f"[risk] Open risk cap exceeded: open={account.open_risk_pct:.2f}%, "
            f"new={proposed_risk_pct:.2f}% -> total={new_total_risk:.2f}% "
            f"(limit={rules.max_total_risk_pct:.2f}%)"
        )
        return False

    return True
