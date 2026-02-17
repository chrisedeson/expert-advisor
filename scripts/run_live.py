#!/usr/bin/env python3
"""Entry point for the live trading engine.

Usage:
    # Demo trading with cTrader (IC Markets)
    python scripts/run_live.py --broker ctrader

    # Paper trading with simulated broker (default)
    python scripts/run_live.py --broker simulated

    # Demo trading - balanced profile
    python scripts/run_live.py --broker ctrader --profile balanced

    # Custom capital
    python scripts/run_live.py --broker ctrader --profile conservative --capital 200

    # Show status
    python scripts/run_live.py --status

    # List available profiles
    python scripts/run_live.py --list-profiles
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

from src.live import LiveEngine, SimulatedBroker, RISK_PROFILES, INSTRUMENTS


def setup_logging(log_dir: str = "logs"):
    """Setup logging to file and console."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"live_{timestamp}.log"

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
    print("\nAvailable risk profiles:")
    print(f"{'Profile':<15} {'Base Lot':<10} {'Mult':<6} {'Description'}")
    print("-" * 60)
    for name, profile in RISK_PROFILES.items():
        print(f"{name:<15} {profile['base_lot']:<10} {profile['mult']:<6} {profile['description']}")
    print(f"\nInstruments: {', '.join(INSTRUMENTS.keys())}")
    print(f"Session: 12-16 UTC (London/NY overlap)")


def create_broker(broker_type: str, capital: float, data_dir: str, is_live: bool = False):
    """Create the appropriate broker instance."""
    if broker_type == "ctrader":
        from src.live.ctrader_broker import CTraderBroker
        return CTraderBroker(is_live=is_live)
    else:
        return SimulatedBroker(
            initial_balance=capital,
            data_dir=data_dir,
        )


def main():
    parser = argparse.ArgumentParser(description='Expert Advisor Live Trading Engine')
    parser.add_argument('--broker', type=str, default='simulated',
                       choices=['simulated', 'ctrader'],
                       help='Broker to use (default: simulated)')
    parser.add_argument('--profile', type=str, default='conservative',
                       choices=list(RISK_PROFILES.keys()),
                       help='Risk profile (default: conservative)')
    parser.add_argument('--capital', type=float, default=500.0,
                       help='Initial capital in USD (default: 500)')
    parser.add_argument('--live', action='store_true', default=False,
                       help='Use LIVE account (default: demo/paper)')
    parser.add_argument('--status', action='store_true',
                       help='Show current status and exit')
    parser.add_argument('--list-profiles', action='store_true',
                       help='List available risk profiles')
    parser.add_argument('--state-dir', type=str, default='state',
                       help='Directory for state persistence')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory for price data')

    args = parser.parse_args()

    if args.list_profiles:
        list_profiles()
        return

    log_file = setup_logging()
    logger = logging.getLogger('main')

    is_paper = args.broker == "simulated" or not args.live
    broker_label = f"cTrader {'LIVE' if args.live else 'DEMO'}" if args.broker == "ctrader" else "Simulated"

    logger.info("=" * 60)
    logger.info("EXPERT ADVISOR - LIVE TRADING ENGINE")
    logger.info("=" * 60)
    logger.info(f"Broker: {broker_label}")
    logger.info(f"Profile: {args.profile}")
    logger.info(f"Capital: ${args.capital}")
    logger.info(f"Paper mode: {is_paper}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"State dir: {args.state_dir}")

    # Create broker
    broker = create_broker(args.broker, args.capital, args.data_dir, args.live)

    # Create engine
    engine = LiveEngine(
        broker=broker,
        risk_profile=args.profile,
        initial_capital=args.capital,
        state_dir=args.state_dir,
        paper_mode=is_paper,
    )

    if args.status:
        broker.connect()
        status = engine.status()
        print(f"\nEngine Status:")
        for k, v in status.items():
            print(f"  {k}: {v}")
        broker.disconnect()
        return

    # Start trading
    print(f"\nStarting {args.profile} trading with ${args.capital}...")
    print(f"Broker: {broker_label}")
    print(f"Instruments: {', '.join(INSTRUMENTS.keys())}")
    print(f"Session: 12-16 UTC only")
    print(f"Press Ctrl+C to stop\n")

    engine.start()


if __name__ == '__main__':
    main()
