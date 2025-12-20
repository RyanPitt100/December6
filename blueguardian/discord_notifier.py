#!/usr/bin/env python3
"""
discord_notifier.py

Discord webhook notifications for trading bot.
Sends formatted alerts to Discord channels for remote monitoring.
"""

import requests
from datetime import datetime
from typing import Optional, Dict, Any
from config_loader import DiscordConfig


class DiscordNotifier:
    """Manages Discord webhook notifications."""

    def __init__(self, config: DiscordConfig):
        self.config = config
        self.session = requests.Session()

    def _send_webhook(self, webhook_url: str, embed: Dict[str, Any]) -> bool:
        """Send embed to Discord webhook with error handling."""
        try:
            payload = {"embeds": [embed]}
            response = self.session.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"[WARNING] Failed to send Discord notification: {e}")
            return False

    def notify_trade_opened(
        self,
        instrument: str,
        direction: str,
        entry: float,
        sl: float,
        tp: Optional[float],
        size_lots: float,
        reason: str
    ) -> bool:
        """Send trade opened notification."""
        if not self.config.alerts.trade_opens:
            return False

        color = 0x00FF00 if direction.lower() == "long" else 0xFF0000  # Green for long, red for short

        embed = {
            "title": f"🟢 Trade Opened: {instrument} {direction.upper()}",
            "color": color,
            "fields": [
                {"name": "Entry", "value": f"{entry:.5f}", "inline": True},
                {"name": "Stop Loss", "value": f"{sl:.5f}", "inline": True},
                {"name": "Take Profit", "value": f"{tp:.5f}" if tp else "None", "inline": True},
                {"name": "Size", "value": f"{size_lots:.2f} lots", "inline": True},
                {"name": "Reason", "value": reason, "inline": True},
            ],
            "footer": {"text": self.config.bot_name},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_webhook(self.config.webhook_url, embed)

    def notify_trade_closed(
        self,
        instrument: str,
        direction: str,
        entry: float,
        exit_price: float,
        pnl: float,
        r_multiple: Optional[float],
        exit_reason: str,
        duration_min: Optional[int]
    ) -> bool:
        """Send trade closed notification."""
        if not self.config.alerts.trade_closes:
            return False

        color = 0x00FF00 if pnl > 0 else 0xFF0000  # Green for profit, red for loss
        pnl_sign = "+" if pnl >= 0 else ""
        r_str = f"{r_multiple:.2f}R" if r_multiple else "N/A"

        embed = {
            "title": f"🔵 Trade Closed: {instrument} {direction.upper()}",
            "color": color,
            "fields": [
                {"name": "Entry", "value": f"{entry:.5f}", "inline": True},
                {"name": "Exit", "value": f"{exit_price:.5f}", "inline": True},
                {"name": "P&L", "value": f"{pnl_sign}${pnl:,.2f} ({r_str})", "inline": True},
                {"name": "Exit Reason", "value": exit_reason, "inline": True},
                {"name": "Duration", "value": f"{duration_min} min" if duration_min else "N/A", "inline": True},
            ],
            "footer": {"text": self.config.bot_name},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_webhook(self.config.webhook_url, embed)

    def notify_risk_warning(self, level: str, message: str, details: Optional[str] = None) -> bool:
        """Send risk warning notification."""
        if not self.config.alerts.risk_warnings:
            return False

        # Color based on severity
        colors = {
            "info": 0x0099FF,      # Blue
            "warning": 0xFFAA00,   # Orange
            "urgent": 0xFF6600,    # Dark orange
            "critical": 0xFF0000,  # Red
        }
        color = colors.get(level.lower(), 0xFFAA00)

        icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "urgent": "🚨",
            "critical": "🛑",
        }
        icon = icons.get(level.lower(), "⚠️")

        embed = {
            "title": f"{icon} Risk {level.upper()}: {message}",
            "color": color,
            "description": details if details else None,
            "footer": {"text": self.config.bot_name},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_webhook(self.config.webhook_url, embed)

    def notify_error(self, category: str, message: str, details: Optional[str] = None) -> bool:
        """Send error notification."""
        if not self.config.alerts.errors:
            return False

        embed = {
            "title": f"❌ Error: {category}",
            "color": 0xFF0000,  # Red
            "description": message,
            "fields": [
                {"name": "Details", "value": details[:1000] if details else "No details", "inline": False}
            ],
            "footer": {"text": self.config.bot_name},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_webhook(self.config.webhook_url, embed)

    def notify_bot_status(self, status: str, message: str) -> bool:
        """Send bot status notification (startup, shutdown)."""
        colors = {
            "startup": 0x00FF00,   # Green
            "shutdown": 0x808080,  # Gray
            "error": 0xFF0000,     # Red
        }
        color = colors.get(status.lower(), 0x0099FF)

        icons = {
            "startup": "🚀",
            "shutdown": "🛑",
            "error": "💥",
        }
        icon = icons.get(status.lower(), "ℹ️")

        embed = {
            "title": f"{icon} Bot {status.title()}",
            "description": message,
            "color": color,
            "footer": {"text": self.config.bot_name},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_webhook(self.config.webhook_url, embed)

    def notify_daily_summary(
        self,
        balance: float,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        trades_today: int,
        open_positions: int,
        stats: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send daily summary to dedicated channel."""
        if not self.config.alerts.daily_summary:
            return False

        pnl_sign = "+" if daily_pnl >= 0 else ""
        color = 0x00FF00 if daily_pnl >= 0 else 0xFF0000

        fields = [
            {"name": "Balance", "value": f"${balance:,.2f}", "inline": True},
            {"name": "Equity", "value": f"${equity:,.2f}", "inline": True},
            {"name": "Daily P&L", "value": f"{pnl_sign}${daily_pnl:,.2f} ({pnl_sign}{daily_pnl_pct:.2f}%)", "inline": True},
            {"name": "Trades Today", "value": str(trades_today), "inline": True},
            {"name": "Open Positions", "value": str(open_positions), "inline": True},
        ]

        if stats:
            fields.extend([
                {"name": "Win Rate", "value": f"{stats.get('win_rate', 0):.1f}%", "inline": True},
                {"name": "Total Trades", "value": str(stats.get('total_trades', 0)), "inline": True},
                {"name": "Profit Factor", "value": f"{stats.get('profit_factor', 0):.2f}", "inline": True},
            ])

        embed = {
            "title": f"📊 Daily Summary - {datetime.now().strftime('%Y-%m-%d')}",
            "color": color,
            "fields": fields,
            "footer": {"text": self.config.bot_name},
            "timestamp": datetime.utcnow().isoformat()
        }

        return self._send_webhook(self.config.daily_summary_webhook_url, embed)


# Global notifier instance
_notifier: Optional[DiscordNotifier] = None


def get_discord_notifier(config: Optional[DiscordConfig] = None) -> Optional[DiscordNotifier]:
    """Get or create the global Discord notifier instance."""
    global _notifier

    if config is None:
        from config_loader import load_discord_config
        config = load_discord_config()

    if config is None or not config.enabled:
        return None

    if _notifier is None:
        _notifier = DiscordNotifier(config)

    return _notifier
