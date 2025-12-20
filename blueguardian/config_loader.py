# config_loader.py
"""
Configuration loader for portfolio and risk management.

Loads YAML configuration files and provides strongly-typed dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import yaml


# ============================================================================
#                         DATACLASS DEFINITIONS
# ============================================================================

@dataclass
class InstrumentConfig:
    """Configuration for a single instrument in the portfolio."""
    symbol: str
    weight: float

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Instrument weight must be between 0 and 1, got {self.weight}")


@dataclass
class InstrumentPortfolioConfig:
    """Configuration for the overall instrument portfolio."""
    portfolio_id: str
    instruments: List[InstrumentConfig]
    notes: str = ""

    def __post_init__(self):
        """Validate portfolio configuration."""
        if not self.instruments:
            raise ValueError("Portfolio must contain at least one instrument")

        total_weight = sum(inst.weight for inst in self.instruments)
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight:.4f}")

    def get_instrument_weight(self, symbol: str) -> float:
        """Get weight for a specific instrument."""
        for inst in self.instruments:
            if inst.symbol == symbol:
                return inst.weight
        return 0.0

    def get_instruments(self) -> List[str]:
        """Get list of instrument symbols."""
        return [inst.symbol for inst in self.instruments]


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing methodology."""
    mode: str  # "atr", "fixed_fraction", etc.
    atr_lookback: int = 20
    atr_multiplier_sl: float = 2.0
    risk_per_trade_basis: str = "equity"  # "equity" or "balance"

    def __post_init__(self):
        """Validate position sizing configuration."""
        valid_modes = ["atr", "fixed_fraction"]
        if self.mode not in valid_modes:
            raise ValueError(f"Position sizing mode must be one of {valid_modes}, got {self.mode}")

        valid_basis = ["equity", "balance"]
        if self.risk_per_trade_basis not in valid_basis:
            raise ValueError(f"Risk basis must be one of {valid_basis}, got {self.risk_per_trade_basis}")


@dataclass
class PortfolioRiskConfig:
    """Configuration for portfolio-level risk parameters."""
    portfolio_id: str
    risk_per_trade_eval_pct: float
    risk_per_trade_funded_pct: float
    max_open_trades: int
    max_risk_at_once_pct: float
    slippage_buffer_pct_of_r: float
    position_sizing: PositionSizingConfig

    def __post_init__(self):
        """Validate portfolio risk configuration."""
        if self.risk_per_trade_eval_pct <= 0:
            raise ValueError("risk_per_trade_eval_pct must be positive")
        if self.risk_per_trade_funded_pct <= 0:
            raise ValueError("risk_per_trade_funded_pct must be positive")
        if self.max_open_trades <= 0:
            raise ValueError("max_open_trades must be positive")
        if self.max_risk_at_once_pct <= 0:
            raise ValueError("max_risk_at_once_pct must be positive")

    def get_risk_per_trade(self, mode: str) -> float:
        """Get risk per trade based on mode (eval or funded)."""
        if mode == "eval":
            return self.risk_per_trade_eval_pct
        elif mode == "funded":
            return self.risk_per_trade_funded_pct
        else:
            raise ValueError(f"Mode must be 'eval' or 'funded', got {mode}")


@dataclass
class RiskEnvelopeGuardConfig:
    """Configuration for risk envelope guard."""
    enabled: bool
    safety_margin_pct: float

    def __post_init__(self):
        """Validate risk envelope guard configuration."""
        if self.safety_margin_pct < 0:
            raise ValueError("safety_margin_pct must be non-negative")


@dataclass
class EvalPhaseConfig:
    """Configuration for FTMO evaluation phase profit target."""
    enabled: bool
    starting_equity: float
    profit_target_pct: float
    lock_trading_on_target: bool

    def __post_init__(self):
        """Validate eval phase configuration."""
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if self.profit_target_pct < 0:
            raise ValueError("profit_target_pct must be non-negative")


@dataclass
class FTMOLimitsConfig:
    """Configuration for prop firm loss limits (FTMO, Blueguardian, etc.)."""
    raw_daily_loss_limit_pct: float
    raw_total_loss_limit_pct: float
    internal_daily_loss_limit_pct: float
    internal_total_loss_limit_pct: float
    # Unrealized loss limits (Blueguardian specific, 0.0 = disabled for FTMO)
    raw_max_unrealised_loss_pct: float = 0.0
    internal_max_unrealised_loss_pct: float = 0.0

    def __post_init__(self):
        """Validate prop firm limits."""
        if self.raw_daily_loss_limit_pct <= 0:
            raise ValueError("raw_daily_loss_limit_pct must be positive")
        if self.raw_total_loss_limit_pct <= 0:
            raise ValueError("raw_total_loss_limit_pct must be positive")
        if self.internal_daily_loss_limit_pct <= 0:
            raise ValueError("internal_daily_loss_limit_pct must be positive")
        if self.internal_total_loss_limit_pct <= 0:
            raise ValueError("internal_total_loss_limit_pct must be positive")

        # Internal limits should be stricter than raw limits
        if self.internal_daily_loss_limit_pct > self.raw_daily_loss_limit_pct:
            raise ValueError("internal_daily_loss_limit_pct must be <= raw_daily_loss_limit_pct")
        if self.internal_total_loss_limit_pct > self.raw_total_loss_limit_pct:
            raise ValueError("internal_total_loss_limit_pct must be <= raw_total_loss_limit_pct")

        # Validate unrealized loss limits if enabled
        if self.raw_max_unrealised_loss_pct > 0:
            if self.internal_max_unrealised_loss_pct > self.raw_max_unrealised_loss_pct:
                raise ValueError("internal_max_unrealised_loss_pct must be <= raw_max_unrealised_loss_pct")


@dataclass
class FTMOActionsConfig:
    """Configuration for FTMO actions on limit breaches."""
    on_soft_brake: str  # e.g., "block_new_trades"
    on_hard_brake: str  # e.g., "flatten_and_disable_until_next_day"

    def __post_init__(self):
        """Validate FTMO actions."""
        valid_soft_actions = ["block_new_trades", "warn"]
        valid_hard_actions = ["flatten_and_disable_until_next_day", "flatten", "halt"]

        if self.on_soft_brake not in valid_soft_actions:
            raise ValueError(f"on_soft_brake must be one of {valid_soft_actions}, got {self.on_soft_brake}")
        if self.on_hard_brake not in valid_hard_actions:
            raise ValueError(f"on_hard_brake must be one of {valid_hard_actions}, got {self.on_hard_brake}")


@dataclass
class FTMOOverlayConfig:
    """Configuration for FTMO overlay rules."""
    account_type: str
    safety_buffer_pct: float
    limits: FTMOLimitsConfig
    risk_envelope_guard: RiskEnvelopeGuardConfig
    actions: FTMOActionsConfig
    eval_phase: Optional[EvalPhaseConfig] = None

    def __post_init__(self):
        """Validate FTMO overlay configuration."""
        if not 0 <= self.safety_buffer_pct <= 1:
            raise ValueError(f"safety_buffer_pct must be between 0 and 1, got {self.safety_buffer_pct}")


@dataclass
class DiscordAlertSettings:
    """Settings for Discord alert types."""
    trade_opens: bool = True
    trade_closes: bool = True
    risk_warnings: bool = True
    errors: bool = True
    daily_summary: bool = True
    daily_summary_time: str = "17:00"


@dataclass
class DiscordConfig:
    """Configuration for Discord webhook notifications."""
    enabled: bool
    bot_name: str
    webhook_url: str
    daily_summary_webhook_url: str
    alerts: DiscordAlertSettings

    def __post_init__(self):
        """Validate Discord configuration."""
        if self.enabled:
            if not self.webhook_url.startswith("https://discord.com"):
                raise ValueError("Invalid Discord main webhook URL")
            if not self.daily_summary_webhook_url.startswith("https://discord.com"):
                raise ValueError("Invalid Discord daily summary webhook URL")


# ============================================================================
#                         YAML LOADING FUNCTIONS
# ============================================================================

def load_instrument_portfolio(file_path: Path) -> InstrumentPortfolioConfig:
    """
    Load instrument portfolio configuration from YAML.

    Args:
        file_path: Path to instrument_portfolio.yml

    Returns:
        InstrumentPortfolioConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML structure is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Instrument portfolio config not found: {file_path}")

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    # Parse instruments
    instruments = []
    for inst_data in data.get("instruments", []):
        instruments.append(InstrumentConfig(
            symbol=inst_data["symbol"],
            weight=inst_data["weight"]
        ))

    return InstrumentPortfolioConfig(
        portfolio_id=data["portfolio_id"],
        instruments=instruments,
        notes=data.get("notes", "")
    )


def load_portfolio_risk_settings(file_path: Path) -> PortfolioRiskConfig:
    """
    Load portfolio risk settings from YAML.

    Args:
        file_path: Path to portfolio_risk_settings.yml

    Returns:
        PortfolioRiskConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML structure is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Portfolio risk settings not found: {file_path}")

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    # Parse position sizing config
    pos_sizing_data = data.get("position_sizing", {})
    position_sizing = PositionSizingConfig(
        mode=pos_sizing_data.get("mode", "atr"),
        atr_lookback=pos_sizing_data.get("atr_lookback", 20),
        atr_multiplier_sl=pos_sizing_data.get("atr_multiplier_sl", 2.0),
        risk_per_trade_basis=pos_sizing_data.get("risk_per_trade_basis", "equity")
    )

    return PortfolioRiskConfig(
        portfolio_id=data["portfolio_id"],
        risk_per_trade_eval_pct=data["risk_per_trade_eval_pct"],
        risk_per_trade_funded_pct=data["risk_per_trade_funded_pct"],
        max_open_trades=data["max_open_trades"],
        max_risk_at_once_pct=data["max_risk_at_once_pct"],
        slippage_buffer_pct_of_r=data["slippage_buffer_pct_of_r"],
        position_sizing=position_sizing
    )


def load_ftmo_overlays(file_path: Path) -> FTMOOverlayConfig:
    """
    Load FTMO overlay configuration from YAML.

    Args:
        file_path: Path to ftmo_overlays.yml

    Returns:
        FTMOOverlayConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML structure is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"FTMO overlays config not found: {file_path}")

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    # Parse limits
    limits_data = data.get("limits", {})
    limits = FTMOLimitsConfig(
        raw_daily_loss_limit_pct=limits_data["raw_daily_loss_limit_pct"],
        raw_total_loss_limit_pct=limits_data["raw_total_loss_limit_pct"],
        internal_daily_loss_limit_pct=limits_data["internal_daily_loss_limit_pct"],
        internal_total_loss_limit_pct=limits_data["internal_total_loss_limit_pct"],
        raw_max_unrealised_loss_pct=limits_data.get("raw_max_unrealised_loss_pct", 0.0),
        internal_max_unrealised_loss_pct=limits_data.get("internal_max_unrealised_loss_pct", 0.0),
    )

    # Parse risk envelope guard
    guard_data = data.get("risk_envelope_guard", {})
    risk_envelope_guard = RiskEnvelopeGuardConfig(
        enabled=guard_data.get("enabled", True),
        safety_margin_pct=guard_data.get("safety_margin_pct", 0.5)
    )

    # Parse actions
    actions_data = data.get("actions", {})
    actions = FTMOActionsConfig(
        on_soft_brake=actions_data.get("on_soft_brake", "block_new_trades"),
        on_hard_brake=actions_data.get("on_hard_brake", "flatten_and_disable_until_next_day")
    )

    # Parse eval_phase (optional)
    eval_phase = None
    eval_phase_data = data.get("eval_phase", {})
    if eval_phase_data and eval_phase_data.get("enabled", False):
        eval_phase = EvalPhaseConfig(
            enabled=eval_phase_data.get("enabled", True),
            starting_equity=eval_phase_data.get("starting_equity", 200000),
            profit_target_pct=eval_phase_data.get("profit_target_pct", 10.0),
            lock_trading_on_target=eval_phase_data.get("lock_trading_on_target", True)
        )

    return FTMOOverlayConfig(
        account_type=data.get("account_type", "FTMO"),
        safety_buffer_pct=data.get("safety_buffer_pct", 0.8),
        limits=limits,
        risk_envelope_guard=risk_envelope_guard,
        actions=actions,
        eval_phase=eval_phase
    )


def load_all_configs(
    base_path: Path | str = "./",
    prop_firm: str = "ftmo"
) -> Tuple[InstrumentPortfolioConfig, PortfolioRiskConfig, FTMOOverlayConfig]:
    """
    Load all configuration files.

    Args:
        base_path: Base directory containing config files (default: current directory)
        prop_firm: Prop firm to load rules for ("ftmo" or "blueguardian")

    Returns:
        Tuple of (InstrumentPortfolioConfig, PortfolioRiskConfig, FTMOOverlayConfig)

    Raises:
        FileNotFoundError: If any config file is missing
        ValueError: If any config is invalid

    Example:
        >>> portfolio_cfg, risk_cfg, ftmo_cfg = load_all_configs(prop_firm="blueguardian")
        >>> print(ftmo_cfg.account_type)
        Blueguardian
    """
    base = Path(base_path)

    portfolio_cfg = load_instrument_portfolio(base / "instrument_portfolio.yml")
    risk_cfg = load_portfolio_risk_settings(base / "portfolio_risk_settings.yml")

    # Load prop firm specific overlay config
    prop_firm_lower = prop_firm.lower()
    if prop_firm_lower == "blueguardian":
        overlay_file = base / "blueguardian_overlays.yml"
    else:
        overlay_file = base / "ftmo_overlays.yml"

    ftmo_cfg = load_ftmo_overlays(overlay_file)
    print(f"[load_all_configs] Loaded prop firm rules: {ftmo_cfg.account_type}")

    # Validate portfolio IDs match
    if portfolio_cfg.portfolio_id != risk_cfg.portfolio_id:
        raise ValueError(
            f"Portfolio ID mismatch: portfolio={portfolio_cfg.portfolio_id}, "
            f"risk={risk_cfg.portfolio_id}"
        )

    return portfolio_cfg, risk_cfg, ftmo_cfg


def load_discord_config(file_path: Path = Path("config/credentials.yml")) -> Optional[DiscordConfig]:
    """
    Load Discord configuration from credentials.yml (optional).

    Args:
        file_path: Path to credentials.yml file

    Returns:
        DiscordConfig instance if Discord is configured and enabled, None otherwise

    Example:
        >>> discord_cfg = load_discord_config()
        >>> if discord_cfg and discord_cfg.enabled:
        ...     print(f"Discord bot name: {discord_cfg.bot_name}")
    """
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    discord_data = config.get('discord', {})
    if not discord_data or not discord_data.get('enabled', False):
        return None

    alerts_data = discord_data.get('alerts', {})
    alerts = DiscordAlertSettings(
        trade_opens=alerts_data.get('trade_opens', True),
        trade_closes=alerts_data.get('trade_closes', True),
        risk_warnings=alerts_data.get('risk_warnings', True),
        errors=alerts_data.get('errors', True),
        daily_summary=alerts_data.get('daily_summary', True),
        daily_summary_time=alerts_data.get('daily_summary_time', '17:00'),
    )

    return DiscordConfig(
        enabled=discord_data['enabled'],
        bot_name=discord_data.get('bot_name', 'Trading Bot'),
        webhook_url=discord_data['webhook_url'],
        daily_summary_webhook_url=discord_data['daily_summary_webhook_url'],
        alerts=alerts,
    )


# ============================================================================
#                         HELPER FUNCTIONS
# ============================================================================

def validate_configs(
    portfolio_cfg: InstrumentPortfolioConfig,
    risk_cfg: PortfolioRiskConfig,
    ftmo_cfg: FTMOOverlayConfig
) -> None:
    """
    Validate loaded configurations for consistency.

    Args:
        portfolio_cfg: Portfolio configuration
        risk_cfg: Risk configuration
        ftmo_cfg: FTMO configuration

    Raises:
        ValueError: If configurations are inconsistent
    """
    # Check risk per trade doesn't exceed max risk at once
    if risk_cfg.risk_per_trade_eval_pct > risk_cfg.max_risk_at_once_pct:
        raise ValueError(
            f"risk_per_trade_eval_pct ({risk_cfg.risk_per_trade_eval_pct}) exceeds "
            f"max_risk_at_once_pct ({risk_cfg.max_risk_at_once_pct})"
        )

    if risk_cfg.risk_per_trade_funded_pct > risk_cfg.max_risk_at_once_pct:
        raise ValueError(
            f"risk_per_trade_funded_pct ({risk_cfg.risk_per_trade_funded_pct}) exceeds "
            f"max_risk_at_once_pct ({risk_cfg.max_risk_at_once_pct})"
        )

    # Check max risk doesn't exceed FTMO daily limit
    if risk_cfg.max_risk_at_once_pct >= ftmo_cfg.limits.internal_daily_loss_limit_pct:
        raise ValueError(
            f"max_risk_at_once_pct ({risk_cfg.max_risk_at_once_pct}) must be < "
            f"internal_daily_loss_limit_pct ({ftmo_cfg.limits.internal_daily_loss_limit_pct})"
        )

    print(f"[validate_configs] All configurations validated successfully")
