#!/usr/bin/env python3
"""
Run Robustness Tests (Phase 6)

Comprehensive testing suite:
1. Multi-pair validation (all 4 pairs)
2. Monte Carlo simulations (10,000 runs)
3. Transaction cost stress testing
4. Parameter sensitivity analysis
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.strategies import EnhancedTrendFollowingStrategy
from src.optimization import RobustnessAnalyzer, MonteCarloSimulator
from src.backtesting import BacktestEngine


def load_all_pairs() -> dict:
    """Load data for all currency pairs."""
    data_dir = Path(__file__).parent.parent / "data" / "raw"

    pairs = {
        'EURUSD': 'EURUSDm_H4.parquet',
        'GBPUSD': 'GBPUSDm_H4.parquet',
        'USDJPY': 'USDJPYm_H4.parquet',
        'AUDUSD': 'AUDUSDm_H4.parquet',
    }

    data_dict = {}
    for symbol, filename in pairs.items():
        filepath = data_dir / filename
        if filepath.exists():
            data_dict[symbol] = pd.read_parquet(filepath)
            logger.info(f"Loaded {symbol}: {len(data_dict[symbol])} candles")
        else:
            logger.warning(f"File not found: {filepath}")

    return data_dict


def main():
    """Run comprehensive robustness tests."""

    logger.info("=" * 70)
    logger.info("PHASE 6: ROBUSTNESS & MONTE CARLO TESTING")
    logger.info("=" * 70)

    # Load all pairs
    logger.info("\n📁 Loading data for all pairs...")
    data_dict = load_all_pairs()

    if len(data_dict) == 0:
        logger.error("No data files found!")
        return

    logger.info(f"Loaded {len(data_dict)} pairs: {list(data_dict.keys())}")

    # Initialize strategy with optimized parameters from Phase 4
    logger.info("\n🔧 Initializing Enhanced Strategy...")
    strategy = EnhancedTrendFollowingStrategy(
        fast_ma_period=40,
        slow_ma_period=186,
        adx_period=15,
        adx_threshold=20.55,
        atr_period=15,
        use_session_filter=False,
        ma_type='ema',
        volatility_lookback=100,
        volatility_threshold=2.5,
        enable_regime_filter=True,
    )

    logger.info(f"Strategy: {strategy.name}")

    # ============================================
    # TEST 1: MULTI-PAIR VALIDATION
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: MULTI-PAIR VALIDATION")
    logger.info("=" * 70)

    analyzer = RobustnessAnalyzer(
        strategy=strategy,
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.15,
    )

    multi_pair_results = analyzer.test_multi_pair(data_dict)

    # ============================================
    # TEST 2: TRANSACTION COST STRESS TEST
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: TRANSACTION COST STRESS TEST")
    logger.info("=" * 70)
    logger.info("Testing with 1x, 1.5x, 2x, 3x normal costs...")

    cost_results = analyzer.test_cost_sensitivity(
        data=data_dict['EURUSD'],
        symbol='EURUSD'
    )

    # ============================================
    # TEST 3: MONTE CARLO SIMULATION
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: MONTE CARLO SIMULATION")
    logger.info("=" * 70)

    # Run backtest to get trades for Monte Carlo
    logger.info("Running baseline backtest to extract trades...")

    signals = strategy.generate_signals(data_dict['EURUSD'])
    engine = BacktestEngine(
        initial_capital=100.0,
        risk_per_trade=0.01,
        max_drawdown=0.15,
    )

    baseline_results = engine.run(
        data=data_dict['EURUSD'],
        signals=signals,
        symbol='EURUSD',
    )

    trades = baseline_results['trades']

    if len(trades) < 20:
        logger.warning(f"Only {len(trades)} trades available. Need at least 20 for reliable Monte Carlo.")
    else:
        logger.info(f"Running Monte Carlo with {len(trades)} trades...")

        mc_simulator = MonteCarloSimulator(
            trades=trades,
            initial_capital=100.0,
        )

        mc_results = mc_simulator.run_simulations(
            n_simulations=10000,
            method='shuffle',
            random_seed=42,
        )

        mc_report = mc_simulator.generate_report()
        logger.info("\n" + mc_report)

    # ============================================
    # TEST 4: PARAMETER SENSITIVITY (OPTIONAL)
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: PARAMETER SENSITIVITY")
    logger.info("=" * 70)
    logger.info("Testing ADX threshold sensitivity...")

    sensitivity_results = analyzer.test_parameter_sensitivity(
        data=data_dict['EURUSD'],
        param_name='adx_threshold',
        param_range=[15.0, 20.0, 25.0, 30.0, 35.0],
        symbol='EURUSD',
    )

    # ============================================
    # GENERATE COMPREHENSIVE REPORT
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("COMPREHENSIVE ROBUSTNESS REPORT")
    logger.info("=" * 70)

    robustness_report = analyzer.generate_report()
    logger.info("\n" + robustness_report)

    # ============================================
    # SAVE RESULTS
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)

    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save multi-pair results
    pair_results_file = output_dir / "multi_pair_results.csv"
    pair_data = []
    for symbol, result in multi_pair_results.items():
        metrics = result['metrics']
        pair_data.append({
            'symbol': symbol,
            'return': metrics['total_return'],
            'cagr': metrics['cagr'],
            'sharpe': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'num_trades': metrics['num_trades'],
            'win_rate': metrics.get('win_rate', 0),
        })

    pd.DataFrame(pair_data).to_csv(pair_results_file, index=False)
    logger.info(f"💾 Multi-pair results: {pair_results_file}")

    # Save Monte Carlo results
    if len(trades) >= 20:
        mc_results_file = output_dir / "monte_carlo_summary.txt"
        with open(mc_results_file, 'w') as f:
            f.write(mc_report)
        logger.info(f"💾 Monte Carlo report: {mc_results_file}")

    # Save comprehensive report
    report_file = output_dir / "robustness_report.txt"
    with open(report_file, 'w') as f:
        f.write(robustness_report)
    logger.info(f"💾 Robustness report: {report_file}")

    # ============================================
    # FINAL VERDICT
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)

    # Check go/no-go criteria
    agg = analyzer.results['multi_pair']['aggregate']

    passed = []
    failed = []

    # Multi-pair criteria
    if agg['positive_pairs'] >= 3:
        passed.append(f"✅ Profitable on ≥3 pairs ({agg['positive_pairs']}/4)")
    else:
        failed.append(f"❌ Profitable on <3 pairs ({agg['positive_pairs']}/4)")

    if agg['avg_sharpe'] > 0.3:
        passed.append(f"✅ Avg Sharpe > 0.3 ({agg['avg_sharpe']:.2f})")
    else:
        failed.append(f"❌ Avg Sharpe < 0.3 ({agg['avg_sharpe']:.2f})")

    # Cost sensitivity criteria
    base_return = cost_results['1.0x']['metrics']['total_return']
    high_cost_return = cost_results['3.0x']['metrics']['total_return']

    if high_cost_return > 0:
        passed.append(f"✅ Still profitable with 3x costs ({high_cost_return*100:.2f}%)")
    else:
        failed.append(f"⚠️  Unprofitable with 3x costs ({high_cost_return*100:.2f}%)")

    # Monte Carlo criteria
    if len(trades) >= 20:
        if mc_results['prob_profit'] > 0.60:
            passed.append(f"✅ P(profit) > 60% ({mc_results['prob_profit']*100:.1f}%)")
        else:
            failed.append(f"❌ P(profit) < 60% ({mc_results['prob_profit']*100:.1f}%)")

    # Display results
    logger.info("")
    for p in passed:
        logger.info(p)
    for f in failed:
        logger.info(f)

    logger.info("")
    logger.info("-" * 70)

    if len(failed) == 0:
        logger.info("🎉 ✅ STRATEGY PASSES ALL ROBUSTNESS TESTS!")
        logger.info("Ready for Phase 7: Analytics & Reporting")
    elif len(failed) <= 2:
        logger.info("⚠️  STRATEGY SHOWS ACCEPTABLE ROBUSTNESS")
        logger.info("Consider minor improvements before live trading")
    else:
        logger.info("❌ STRATEGY NEEDS SIGNIFICANT IMPROVEMENT")
        logger.info("Review failed criteria before proceeding")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
