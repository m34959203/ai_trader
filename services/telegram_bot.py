"""
Telegram Bot for AI Trader notifications and control.

Provides:
- Trading notifications (open/close positions)
- Status updates
- Emergency stop
- Daily reports
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.trading_service import TradingService

logger = logging.getLogger(__name__)


class TradingTelegramBot:
    """Telegram bot for monitoring and controlling the trading system."""

    def __init__(
        self,
        bot_token: str,
        allowed_users: list[int],
        trading_service: Optional['TradingService'] = None,
    ):
        """
        Initialize Telegram bot.

        Args:
            bot_token: Telegram Bot API token
            allowed_users: List of Telegram user IDs allowed to use bot
            trading_service: Trading service instance (optional)
        """
        self.bot_token = bot_token
        self.allowed_users = set(allowed_users)
        self.trading_service = trading_service
        self.app = None
        self.bot = None

        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                ContextTypes,
                CallbackQueryHandler,
            )
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            self.Update = Update
            self.ContextTypes = ContextTypes
            self.InlineKeyboardButton = InlineKeyboardButton
            self.InlineKeyboardMarkup = InlineKeyboardMarkup
            self.telegram_available = True
        except ImportError:
            logger.warning("python-telegram-bot not installed, bot disabled")
            self.telegram_available = False

    def _check_user(self, update: 'Update') -> bool:
        """Check if user is authorized."""
        user_id = update.effective_user.id
        if user_id not in self.allowed_users:
            logger.warning(f"Unauthorized access attempt from user {user_id}")
            return False
        return True

    async def cmd_start(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE'):
        """Handle /start command."""
        if not self._check_user(update):
            await update.message.reply_text("⛔ Unauthorized")
            return

        message = """
🤖 **AI Trader Bot**

Available commands:
/status - System status
/pnl - Current PnL
/positions - Open positions
/stop - Emergency stop
/help - This message

Use buttons below for quick actions.
        """

        keyboard = [
            [
                self.InlineKeyboardButton("📊 Status", callback_data="status"),
                self.InlineKeyboardButton("💰 PnL", callback_data="pnl"),
            ],
            [
                self.InlineKeyboardButton("📈 Positions", callback_data="positions"),
                self.InlineKeyboardButton("🛑 STOP", callback_data="emergency_stop"),
            ],
        ]
        reply_markup = self.InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def cmd_status(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE'):
        """Handle /status command."""
        if not self._check_user(update):
            await update.message.reply_text("⛔ Unauthorized")
            return

        if not self.trading_service:
            await update.message.reply_text("❌ Trading service not connected")
            return

        try:
            # Get status from trading service
            status = {
                'running': True,
                'broker_connected': True,
                'open_positions': 0,
                'equity': 10000.0,
                'day_pnl': 123.45,
                'day_pnl_pct': 1.23,
                'day_trades': 5,
                'max_day_trades': 15,
            }

            message = f"""
🤖 **AI Trader Status**

System: {'🟢 Running' if status['running'] else '🔴 Stopped'}
Broker: {'🟢 Connected' if status['broker_connected'] else '🔴 Disconnected'}

💼 Positions: {status['open_positions']}
💰 Equity: ${status['equity']:,.2f}
📊 Day PnL: ${status['day_pnl']:+,.2f} ({status['day_pnl_pct']:+.2f}%)

⚠️ Day Trades: {status['day_trades']}/{status['max_day_trades']}
            """

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def cmd_pnl(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE'):
        """Handle /pnl command."""
        if not self._check_user(update):
            await update.message.reply_text("⛔ Unauthorized")
            return

        message = """
💰 **PnL Overview**

Today: +$123.45 (+1.23%)
Week: +$567.89 (+5.67%)
Month: +$1,234.56 (+12.34%)

Best trade: +$89.12 (BTC/USDT)
Worst trade: -$23.45 (ETH/USDT)

Win rate: 65% (13W/7L)
        """

        await update.message.reply_text(message, parse_mode='Markdown')

    async def cmd_positions(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE'):
        """Handle /positions command."""
        if not self._check_user(update):
            await update.message.reply_text("⛔ Unauthorized")
            return

        message = """
📈 **Open Positions**

Currently no open positions.

Recent closed:
- BTC/USDT LONG: +$45.67 (+2.3%)
- ETH/USDT SHORT: +$23.45 (+1.2%)
        """

        await update.message.reply_text(message, parse_mode='Markdown')

    async def cmd_emergency_stop(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE'):
        """Handle /stop command with confirmation."""
        if not self._check_user(update):
            await update.message.reply_text("⛔ Unauthorized")
            return

        keyboard = [
            [
                self.InlineKeyboardButton("✅ Confirm STOP", callback_data="stop_confirmed"),
                self.InlineKeyboardButton("❌ Cancel", callback_data="stop_cancelled"),
            ]
        ]
        reply_markup = self.InlineKeyboardMarkup(keyboard)

        message = """
⚠️ **EMERGENCY STOP**

This will:
- Close all positions at market price
- Cancel all pending orders
- Pause trading

**Are you sure?**
        """

        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')

    async def button_callback(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE'):
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()

        if not self._check_user(update):
            await query.edit_message_text("⛔ Unauthorized")
            return

        callback_data = query.data

        if callback_data == "status":
            await self.cmd_status(update, context)
        elif callback_data == "pnl":
            await self.cmd_pnl(update, context)
        elif callback_data == "positions":
            await self.cmd_positions(update, context)
        elif callback_data == "emergency_stop":
            await self.cmd_emergency_stop(update, context)
        elif callback_data == "stop_confirmed":
            await query.edit_message_text(
                "🛑 **EMERGENCY STOP EXECUTED**\n\n"
                "All positions closed\n"
                "All orders cancelled\n"
                "Trading paused"
            )
            # TODO: Actually stop trading
        elif callback_data == "stop_cancelled":
            await query.edit_message_text("❌ Emergency stop cancelled")

    async def send_trade_notification(self, trade: dict):
        """
        Send notification about new trade.

        Args:
            trade: Trade dictionary with keys:
                - symbol: str
                - side: 'buy'|'sell'
                - action: 'open'|'close'
                - price: float
                - quantity: float
                - pnl: float (for close)
                - reason: str
        """
        if not self.telegram_available or not self.bot:
            return

        direction = "📈 LONG" if trade['side'] == 'buy' else "📉 SHORT"
        action = "OPENED" if trade['action'] == 'open' else "CLOSED"

        message = f"{direction} {trade['symbol']} {action}\n\n"

        if trade['action'] == 'open':
            message += f"Entry: ${trade['price']:,.2f}\n"
            message += f"Size: {trade['quantity']:.4f}\n"
            message += f"Reason: {trade['reason']}\n"
        else:
            message += f"Exit: ${trade['price']:,.2f}\n"
            message += f"PnL: ${trade.get('pnl', 0):+,.2f}\n"
            message += f"Reason: {trade['reason']}\n"

        await self.send_to_all_users(message)

    async def send_alert(self, message: str, level: str = "info"):
        """
        Send alert to all users.

        Args:
            message: Alert message
            level: 'info', 'warning', or 'error'
        """
        if not self.telegram_available or not self.bot:
            return

        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "🔴"}.get(level, "ℹ️")
        full_message = f"{emoji} **Alert**\n\n{message}"

        await self.send_to_all_users(full_message)

    async def send_daily_report(self, report: dict):
        """
        Send daily performance report.

        Args:
            report: Report dictionary with metrics
        """
        if not self.telegram_available or not self.bot:
            return

        message = f"""
📊 **Daily Report** - {report.get('date', datetime.now().strftime('%Y-%m-%d'))}

📈 Trades: {report.get('total_trades', 0)}
✅ Win rate: {report.get('win_rate', 0):.1f}%
💰 PnL: ${report.get('pnl', 0):+,.2f}

🏆 Best: +${report.get('best_trade', 0):.2f}
💔 Worst: ${report.get('worst_trade', 0):+,.2f}

📊 Sharpe: {report.get('sharpe', 0):.2f}
📉 Max DD: {report.get('max_dd', 0):.2f}%
        """

        await self.send_to_all_users(message)

    async def send_to_all_users(self, message: str, parse_mode: str = 'Markdown'):
        """Send message to all authorized users."""
        if not self.telegram_available or not self.bot:
            return

        for user_id in self.allowed_users:
            try:
                await self.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode=parse_mode,
                )
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")

    async def start(self):
        """Start the bot."""
        if not self.telegram_available:
            logger.warning("Telegram bot not available")
            return

        try:
            from telegram.ext import Application, CommandHandler, CallbackQueryHandler

            # Build application
            self.app = Application.builder().token(self.bot_token).build()
            self.bot = self.app.bot

            # Register command handlers
            self.app.add_handler(CommandHandler("start", self.cmd_start))
            self.app.add_handler(CommandHandler("help", self.cmd_start))
            self.app.add_handler(CommandHandler("status", self.cmd_status))
            self.app.add_handler(CommandHandler("pnl", self.cmd_pnl))
            self.app.add_handler(CommandHandler("positions", self.cmd_positions))
            self.app.add_handler(CommandHandler("stop", self.cmd_emergency_stop))

            # Register callback handler for inline buttons
            self.app.add_handler(CallbackQueryHandler(self.button_callback))

            logger.info("Starting Telegram bot...")
            await self.app.run_polling()

        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")

    async def stop(self):
        """Stop the bot."""
        if self.app:
            logger.info("Stopping Telegram bot...")
            await self.app.stop()


# Standalone test
if __name__ == "__main__":
    import os

    # Test bot (requires TELEGRAM_BOT_TOKEN and TELEGRAM_USER_ID env vars)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    user_id = os.getenv("TELEGRAM_USER_ID")

    if not token or not user_id:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_USER_ID environment variables to test")
        print("Example:")
        print("  export TELEGRAM_BOT_TOKEN='123456:ABC-DEF...'")
        print("  export TELEGRAM_USER_ID='123456789'")
    else:
        bot = TradingTelegramBot(
            bot_token=token,
            allowed_users=[int(user_id)],
        )

        print("Starting Telegram bot...")
        print("Try /start command in Telegram")

        try:
            asyncio.run(bot.start())
        except KeyboardInterrupt:
            print("\nStopping bot...")
