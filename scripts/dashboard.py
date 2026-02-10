#!/usr/bin/env python3
"""
Smart Grid EA - Monitoring Dashboard

Terminal-based dashboard for quick status checks.

Usage:
    python scripts/dashboard.py          # One-time view
    python scripts/dashboard.py --watch  # Auto-refresh every 10 seconds

Shows:
- Current balance & P&L
- Trading status
- Open positions
- Protection system status
- Recent alerts
- Performance metrics

Goal: Understand everything in 10 seconds
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import time
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


# Terminal colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def green(text): return f"{Colors.OKGREEN}{text}{Colors.ENDC}"

    @staticmethod
    def red(text): return f"{Colors.FAIL}{text}{Colors.ENDC}"

    @staticmethod
    def yellow(text): return f"{Colors.WARNING}{text}{Colors.ENDC}"

    @staticmethod
    def blue(text): return f"{Colors.OKBLUE}{text}{Colors.ENDC}"

    @staticmethod
    def cyan(text): return f"{Colors.OKCYAN}{text}{Colors.ENDC}"

    @staticmethod
    def bold(text): return f"{Colors.BOLD}{text}{Colors.ENDC}"


class Dashboard:
    """Terminal-based monitoring dashboard"""

    def __init__(self, data_dir: Path):
        """Initialize dashboard"""
        self.data_dir = data_dir
        self.status_file = data_dir / "status.json"
        self.state_file = data_dir / "state.json"

    def load_status(self) -> Dict:
        """Load current EA status"""
        # TODO: Load from actual state files
        # For now, return mock data for display testing
        return {
            'timestamp': datetime.now(),
            'balance': 523.45,
            'initial_capital': 100.0,
            'equity': 526.60,
            'status': 'TRADING',  # TRADING, PAUSED, CRISIS, STOPPED
            'positions_count': 2,
            'unrealized_pnl': 3.15,

            # Performance
            'today_pnl': 2.30,
            'today_pnl_pct': 0.4,
            'week_pnl': 15.23,
            'week_pnl_pct': 2.9,
            'month_pnl': 42.10,
            'month_pnl_pct': 8.8,
            'year_pnl': 189.45,
            'year_pnl_pct': 56.7,

            # Protection systems
            'market_condition': 'NORMAL',  # NORMAL, ELEVATED, CRISIS
            'atr_ratio': 0.91,
            'circuit_breaker_state': 'ACTIVE',  # ACTIVE, DAILY_PAUSE, WEEKLY_PAUSE, MONTHLY_STOP

            # Recent activity
            'last_trade_time': datetime.now() - timedelta(hours=2),
            'last_trade_pnl': 5.20,
            'trades_today': 3,
            'trades_week': 12,

            # Alerts
            'unread_alerts': 0,
            'last_alert': None,
        }

    def clear_screen(self):
        """Clear terminal screen"""
        print("\033[H\033[J", end="")

    def render(self):
        """Render dashboard"""
        self.clear_screen()

        status = self.load_status()

        # Header
        self.print_header()

        # Main status
        self.print_main_status(status)

        # Performance
        self.print_performance(status)

        # Protection systems
        self.print_protection_status(status)

        # Recent activity
        self.print_recent_activity(status)

        # Actions
        self.print_actions()

        # Footer
        self.print_footer(status)

    def print_header(self):
        """Print dashboard header"""
        print("╔" + "═" * 78 + "╗")
        print("║" + Colors.bold(" " * 20 + "SMART GRID EA - DASHBOARD" + " " * 33) + "║")
        print("╠" + "═" * 78 + "╣")

    def print_main_status(self, status: Dict):
        """Print main status section"""
        # Status indicator
        status_text = status['status']
        if status_text == 'TRADING':
            status_display = Colors.green("✅ TRADING - NORMAL MODE")
        elif status_text == 'PAUSED':
            status_display = Colors.yellow("⏸️  PAUSED")
        elif status_text == 'CRISIS':
            status_display = Colors.red("🚨 CRISIS MODE")
        else:
            status_display = Colors.red("🛑 STOPPED")

        print(f"║ Status: {status_display}" + " " * (71 - len(status_text) - 8) + "║")

        # Balance
        balance = status['balance']
        equity = status['equity']
        initial = status['initial_capital']

        total_roi = ((balance - initial) / initial) * 100
        balance_color = Colors.green if balance >= initial else Colors.red

        unrealized = status['unrealized_pnl']
        unrealized_color = Colors.green if unrealized >= 0 else Colors.red

        print(f"║ Balance: {balance_color(f'${balance:.2f}')} (▲ {balance_color(f'+${balance - initial:.2f}')} / {balance_color(f'+{total_roi:.1f}%')} total)" + " " * (78 - 45 - len(f"{balance:.2f}") - len(f"{balance - initial:.2f}") - len(f"{total_roi:.1f}")) + "║")
        print(f"║ Open Positions: {status['positions_count']} ({unrealized_color(f'{unrealized:+.2f}')} unrealized)" + " " * (78 - 36 - len(f"{unrealized:+.2f}")) + "║")

        print("╠" + "═" * 78 + "╣")

    def print_performance(self, status: Dict):
        """Print performance metrics"""
        print("║ " + Colors.bold("Performance") + " " * 67 + "║")

        # Helper to format P&L
        def fmt_pnl(pnl, pct):
            if pnl >= 0:
                return Colors.green(f"+${pnl:.2f} (+{pct:.1f}%)")
            else:
                return Colors.red(f"-${abs(pnl):.2f} ({pct:.1f}%)")

        today = fmt_pnl(status['today_pnl'], status['today_pnl_pct'])
        week = fmt_pnl(status['week_pnl'], status['week_pnl_pct'])
        month = fmt_pnl(status['month_pnl'], status['month_pnl_pct'])
        year = fmt_pnl(status['year_pnl'], status['year_pnl_pct'])

        print(f"║   Today:  {today}" + " " * (78 - 12 - len(f"{status['today_pnl']:.2f}") - len(f"{status['today_pnl_pct']:.1f}") - 10) + "║")
        print(f"║   Week:   {week}" + " " * (78 - 12 - len(f"{status['week_pnl']:.2f}") - len(f"{status['week_pnl_pct']:.1f}") - 10) + "║")
        print(f"║   Month:  {month}" + " " * (78 - 12 - len(f"{status['month_pnl']:.2f}") - len(f"{status['month_pnl_pct']:.1f}") - 10) + "║")
        print(f"║   Year:   {year}" + " " * (78 - 12 - len(f"{status['year_pnl']:.2f}") - len(f"{status['year_pnl_pct']:.1f}") - 10) + "║")

        print("╠" + "═" * 78 + "╣")

    def print_protection_status(self, status: Dict):
        """Print protection system status"""
        print("║ " + Colors.bold("Protection Systems") + " " * 59 + "║")

        # Market condition
        condition = status['market_condition']
        atr_ratio = status['atr_ratio']

        if condition == 'NORMAL':
            condition_display = Colors.green(f"NORMAL (ATR: {atr_ratio:.2f}x)")
        elif condition == 'ELEVATED':
            condition_display = Colors.yellow(f"ELEVATED (ATR: {atr_ratio:.2f}x)")
        else:
            condition_display = Colors.red(f"CRISIS (ATR: {atr_ratio:.2f}x)")

        print(f"║   Market Condition: {condition_display}" + " " * (78 - 23 - len(condition) - len(f"{atr_ratio:.2f}") - 9) + "║")

        # Circuit breaker
        cb_state = status['circuit_breaker_state']
        if cb_state == 'ACTIVE':
            cb_display = Colors.green("✅ All limits OK")
        elif cb_state == 'DAILY_PAUSE':
            cb_display = Colors.yellow("⏸️  Daily pause (24h)")
        elif cb_state == 'WEEKLY_PAUSE':
            cb_display = Colors.yellow("⏸️  Weekly pause (7d)")
        else:
            cb_display = Colors.red("🛑 Manual restart required")

        print(f"║   Risk Status: {cb_display}" + " " * (78 - 18 - len(cb_display) + 9) + "║")

        print("╠" + "═" * 78 + "╣")

    def print_recent_activity(self, status: Dict):
        """Print recent activity"""
        print("║ " + Colors.bold("Recent Activity") + " " * 62 + "║")

        if status['last_alert']:
            print(f"║   [ALERT] {status['last_alert']}" + " " * (78 - 12 - len(status['last_alert'])) + "║")
        else:
            print("║   [INFO] Weekly summary sent" + " " * 48 + "║")

        last_trade = status['last_trade_time']
        time_ago = datetime.now() - last_trade
        hours_ago = int(time_ago.total_seconds() / 3600)

        trade_pnl = status['last_trade_pnl']
        trade_color = Colors.green if trade_pnl > 0 else Colors.red

        print(f"║   [INFO] Position closed {trade_color(f'{trade_pnl:+.2f}')} ({hours_ago}h ago)" + " " * (78 - 36 - len(f"{trade_pnl:+.2f}") - len(str(hours_ago))) + "║")

        print("╠" + "═" * 78 + "╣")

    def print_actions(self):
        """Print available actions"""
        print("║ " + Colors.bold("Actions:") + " [R]esume  [P]ause  [C]lose All  [L]ogs  [Q]uit" + " " * 18 + "║")

    def print_footer(self, status: Dict):
        """Print footer"""
        timestamp = status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        print("╠" + "═" * 78 + "╣")
        print(f"║ Last updated: {timestamp}" + " " * (78 - 15 - len(timestamp)) + "║")
        print("╚" + "═" * 78 + "╝")

    def run_watch_mode(self, refresh_seconds: int = 10):
        """Run dashboard in watch mode with auto-refresh"""
        print(Colors.cyan("Dashboard starting in watch mode..."))
        print(Colors.cyan(f"Auto-refresh every {refresh_seconds} seconds"))
        print(Colors.cyan("Press Ctrl+C to exit\n"))
        time.sleep(2)

        try:
            while True:
                self.render()
                time.sleep(refresh_seconds)
        except KeyboardInterrupt:
            print("\n\n" + Colors.cyan("Dashboard stopped."))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart Grid EA Dashboard")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - auto-refresh every 10 seconds"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10)"
    )

    args = parser.parse_args()

    # Data directory
    data_dir = Path(__file__).parent.parent / "data" / "state"
    data_dir.mkdir(parents=True, exist_ok=True)

    dashboard = Dashboard(data_dir)

    if args.watch:
        dashboard.run_watch_mode(refresh_seconds=args.refresh)
    else:
        dashboard.render()


if __name__ == "__main__":
    main()
