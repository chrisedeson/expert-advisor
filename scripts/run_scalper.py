#!/usr/bin/env python3
"""Entry point for the Supply & Demand Scalper EA.

Runs separately from the Grid EA with its own state, logs, and Telegram prefix.

Usage:
    # Demo trading with cTrader
    python scripts/run_scalper.py --broker ctrader

    # Conservative profile (default)
    python scripts/run_scalper.py --broker ctrader --profile conservative --capital 100

    # Balanced profile
    python scripts/run_scalper.py --broker ctrader --profile balanced --capital 200

    # With session filter (London/NY overlap only)
    python scripts/run_scalper.py --broker ctrader --session 12-16

    # Show status
    python scripts/run_scalper.py --broker ctrader --status

    # List available profiles
    python scripts/run_scalper.py --list-profiles
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.scalper.scalper_engine import ScalperEngine, SCALPER_INSTRUMENTS, SCALPER_PROFILES
from src.live.notifier import TelegramNotifier


def setup_logging(log_dir: str = "logs"):
    """Setup logging to file and console."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"scalper_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return log_file


def list_profiles():
    print("\nScalper Risk Profiles:")
    print(f"{'Profile':<15} {'Risk%':<8} {'R:R':<6} {'MaxPos':<8} {'Description'}")
    print("-" * 70)
    for name, p in SCALPER_PROFILES.items():
        print(f"{name:<15} {p['risk_pct']*100:.0f}%{'':<4} {p['fixed_rr']:<6} {p['max_concurrent']:<8} {p['description']}")
    print(f"\nInstruments: {', '.join(SCALPER_INSTRUMENTS.keys())}")
    print(f"Strategy: Supply & Demand zones (H1) with M15 entries")


def create_broker(broker_type: str, is_live: bool = False, account_env: str = None,
                   watch_symbols: list = None):
    """Create the appropriate broker instance."""
    if broker_type == "ctrader":
        from src.live.ctrader_broker import CTraderBroker
        return CTraderBroker(is_live=is_live, account_env=account_env,
                            watch_symbols=watch_symbols)
    else:
        raise ValueError(f"Scalper only supports ctrader broker (got: {broker_type})")


def parse_session(session_str: str):
    """Parse session string like '12-16' into (start, end)."""
    if not session_str or session_str.lower() == '24h':
        return -1, -1
    parts = session_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid session format: {session_str}. Use e.g. '12-16'")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(description='Supply & Demand Scalper EA')
    parser.add_argument('--broker', type=str, default='ctrader',
                       choices=['ctrader'],
                       help='Broker to use (default: ctrader)')
    parser.add_argument('--profile', type=str, default='conservative',
                       choices=list(SCALPER_PROFILES.keys()),
                       help='Risk profile (default: conservative)')
    parser.add_argument('--capital', type=float, default=100.0,
                       help='Initial capital in USD (default: 100)')
    parser.add_argument('--live', action='store_true', default=False,
                       help='Use LIVE account (default: demo)')
    parser.add_argument('--session', type=str, default='24h',
                       help='Trading session hours, e.g. "12-16" or "24h" (default: 24h)')
    parser.add_argument('--status', action='store_true',
                       help='Show current status and exit')
    parser.add_argument('--list-profiles', action='store_true',
                       help='List available risk profiles')
    parser.add_argument('--state-dir', type=str, default='state_scalper',
                       help='Directory for state persistence (default: state_scalper)')
    parser.add_argument('--no-notify', action='store_true', default=False,
                       help='Disable Telegram notifications')
    parser.add_argument('--account-env', type=str, default='CTRADER_SCALPER_ACCOUNT_ID',
                       help='Env var for cTrader account ID (default: CTRADER_SCALPER_ACCOUNT_ID)')
    parser.add_argument('--instruments', type=str, default=None,
                       help='Comma-separated instrument list (default: all)')

    args = parser.parse_args()

    if args.list_profiles:
        list_profiles()
        return

    log_file = setup_logging()
    logger = logging.getLogger('main')

    session_start, session_end = parse_session(args.session)
    broker_label = f"cTrader {'LIVE' if args.live else 'DEMO'}"
    session_label = f"{session_start}-{session_end} UTC" if session_start >= 0 else "24h"

    # Parse instruments
    instruments = None
    if args.instruments:
        syms = [s.strip().upper() for s in args.instruments.split(',')]
        for s in syms:
            if s not in SCALPER_INSTRUMENTS:
                logger.error(f"Unknown instrument: {s}. Available: {list(SCALPER_INSTRUMENTS.keys())}")
                return
        instruments = {s: 1.0 / len(syms) for s in syms}

    logger.info("=" * 60)
    logger.info("SUPPLY & DEMAND SCALPER EA")
    logger.info("=" * 60)
    logger.info(f"Broker: {broker_label}")
    logger.info(f"Profile: {args.profile}")
    logger.info(f"Capital: ${args.capital}")
    logger.info(f"Session: {session_label}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"State dir: {args.state_dir}")
    logger.info(f"Account env: {args.account_env}")

    # Create broker with the scalper's instrument list
    inst_list = list(instruments.keys()) if instruments else list(SCALPER_INSTRUMENTS.keys())
    broker = create_broker(args.broker, args.live, args.account_env,
                           watch_symbols=inst_list)

    # Create notifier
    notifier = None
    if not args.no_notify:
        notifier = TelegramNotifier()
        if notifier.enabled:
            logger.info("Telegram notifications: ENABLED (prefixed [SCALPER])")
        else:
            logger.info("Telegram notifications: DISABLED (missing env vars)")
            notifier = None

    # Create engine
    engine = ScalperEngine(
        broker=broker,
        risk_profile=args.profile,
        instruments=instruments,
        initial_capital=args.capital,
        state_dir=args.state_dir,
        notifier=notifier,
        session_start=session_start,
        session_end=session_end,
    )

    if args.status:
        broker.connect()
        status = engine.status()
        print(f"\nScalper Status:")
        for k, v in status.items():
            print(f"  {k}: {v}")
        broker.disconnect()
        return

    # Start trading
    print(f"\nStarting {args.profile} scalper with ${args.capital}...")
    print(f"Broker: {broker_label}")
    print(f"Instruments: {', '.join(inst_list)}")
    print(f"Session: {session_label}")
    print(f"Strategy: S&D zones (H1) + M15 entries")
    print(f"Risk: {SCALPER_PROFILES[args.profile]['risk_pct']*100:.0f}% per trade, "
          f"R:R {SCALPER_PROFILES[args.profile]['fixed_rr']}")
    print(f"Press Ctrl+C to stop\n")

    engine.start()


if __name__ == '__main__':
    main()
