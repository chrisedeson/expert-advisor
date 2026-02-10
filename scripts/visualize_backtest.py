#!/usr/bin/env python3
"""
Backtest Visualization

Creates charts to visualize backtest results:
1. Equity curve with drawdowns
2. Monthly returns heatmap
3. Protection events timeline
4. Trade distribution
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from loguru import logger

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_equity_curve(result, output_path: Path):
    """Plot equity curve with drawdown shading"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    equity_curve = result.equity_curve

    # Equity curve
    ax1.plot(equity_curve.index, equity_curve['equity'], label='Equity', linewidth=2)
    ax1.plot(equity_curve.index, equity_curve['peak'], label='Peak', linestyle='--', alpha=0.7)
    ax1.axhline(result.initial_balance, color='gray', linestyle=':', alpha=0.5, label='Initial Balance')

    ax1.set_ylabel('Balance ($)', fontsize=12)
    ax1.set_title(
        f'Equity Curve - ${result.initial_balance:.0f} → ${result.final_balance:.0f} '
        f'({result.total_return*100:.1f}% return, {result.cagr*100:.1f}% CAGR)',
        fontsize=14,
        fontweight='bold'
    )
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(
        equity_curve.index,
        equity_curve['drawdown'] * 100,
        0,
        color='red',
        alpha=0.3,
        label='Drawdown'
    )
    ax2.plot(equity_curve.index, equity_curve['drawdown'] * 100, color='darkred', linewidth=1)

    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title(f'Drawdown - Max: {result.max_drawdown*100:.2f}%', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.success(f"Saved equity curve: {output_path}")
    plt.close()


def plot_monthly_returns(result, output_path: Path):
    """Plot monthly returns heatmap"""
    equity_curve = result.equity_curve.copy()

    # Calculate monthly returns
    equity_curve['year'] = equity_curve.index.year
    equity_curve['month'] = equity_curve.index.month

    monthly_returns = equity_curve.groupby(['year', 'month'])['equity'].last().pct_change() * 100

    # Pivot for heatmap
    monthly_pivot = monthly_returns.reset_index()
    monthly_pivot.columns = ['year', 'month', 'return']
    monthly_pivot = monthly_pivot.pivot(index='year', columns='month', values='return')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    # Custom colormap (green for positive, red for negative)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(
        monthly_pivot,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        center=0,
        cbar_kws={'label': 'Return (%)'},
        linewidths=0.5,
        ax=ax
    )

    ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.success(f"Saved monthly returns: {output_path}")
    plt.close()


def plot_protection_events(result, output_path: Path):
    """Plot protection events timeline"""
    if len(result.protection_events) == 0:
        logger.info("No protection events to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot equity curve
    equity_curve = result.equity_curve
    ax.plot(equity_curve.index, equity_curve['equity'], label='Equity', color='blue', linewidth=2, alpha=0.7)

    # Mark protection events
    event_types = {
        'circuit_breaker': ('red', 'o', 'Circuit Breaker'),
        'crisis_mode': ('orange', 's', 'Crisis Mode'),
        'volatility_pause': ('yellow', '^', 'Volatility Pause'),
        'force_close': ('darkred', 'X', 'Force Close'),
        'protection_active': ('gray', '.', 'Protection Active'),
    }

    for event in result.protection_events:
        event_type = event.event_type
        if event_type in event_types:
            color, marker, label = event_types[event_type]
            ax.scatter(
                event.timestamp,
                event.balance_at_event,
                color=color,
                marker=marker,
                s=100,
                alpha=0.7,
                label=label,
                zorder=10
            )

    ax.set_ylabel('Balance ($)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f'Protection Events Timeline ({len(result.protection_events)} events)', fontsize=14, fontweight='bold')

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.success(f"Saved protection events: {output_path}")
    plt.close()


def plot_trade_distribution(result, output_path: Path):
    """Plot trade P&L distribution"""
    if len(result.trades) == 0:
        logger.info("No trades to plot")
        return

    trade_pnls = [t.net_pnl for t in result.trades]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(trade_pnls, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.axvline(np.mean(trade_pnls), color='green', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(trade_pnls):.2f}')

    ax1.set_xlabel('Trade P&L ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(trade_pnls, vert=True, patch_artist=True)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)

    ax2.set_ylabel('Trade P&L ($)', fontsize=12)
    ax2.set_title('Trade P&L Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.success(f"Saved trade distribution: {output_path}")
    plt.close()


def plot_comparison(protected_result, unprotected_result, output_path: Path):
    """Plot comparison between protected and unprotected"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Equity curves
    ax1 = axes[0, 0]
    ax1.plot(protected_result.equity_curve.index, protected_result.equity_curve['equity'],
             label='Protected', linewidth=2, color='green')
    ax1.plot(unprotected_result.equity_curve.index, unprotected_result.equity_curve['equity'],
             label='Unprotected', linewidth=2, color='red', alpha=0.7)
    ax1.set_ylabel('Balance ($)', fontsize=12)
    ax1.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdowns
    ax2 = axes[0, 1]
    ax2.fill_between(protected_result.equity_curve.index,
                     protected_result.equity_curve['drawdown'] * 100, 0,
                     color='green', alpha=0.3, label='Protected')
    ax2.fill_between(unprotected_result.equity_curve.index,
                     unprotected_result.equity_curve['drawdown'] * 100, 0,
                     color='red', alpha=0.3, label='Unprotected')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # 3. Metrics comparison
    ax3 = axes[1, 0]
    metrics = ['Return\n(%)', 'CAGR\n(%)', 'Max DD\n(%)', 'Sharpe', 'Win Rate\n(%)']
    protected_vals = [
        protected_result.total_return * 100,
        protected_result.cagr * 100,
        protected_result.max_drawdown * 100,
        protected_result.sharpe_ratio,
        protected_result.win_rate * 100,
    ]
    unprotected_vals = [
        unprotected_result.total_return * 100,
        unprotected_result.cagr * 100,
        unprotected_result.max_drawdown * 100,
        unprotected_result.sharpe_ratio,
        unprotected_result.win_rate * 100,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax3.bar(x - width/2, protected_vals, width, label='Protected', color='green', alpha=0.7)
    ax3.bar(x + width/2, unprotected_vals, width, label='Unprotected', color='red', alpha=0.7)

    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Protection impact
    ax4 = axes[1, 1]
    impact_labels = ['Circuit\nBreaker', 'Crisis\nMode', 'Volatility\nPause', 'Positions\nClosed']
    impact_values = [
        protected_result.circuit_breaker_triggers,
        protected_result.crisis_mode_activations,
        protected_result.volatility_pauses,
        protected_result.positions_force_closed,
    ]

    colors = ['red', 'orange', 'yellow', 'darkred']
    ax4.bar(impact_labels, impact_values, color=colors, alpha=0.7)

    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title(f'Protection System Activations\n(Capital Saved: ${protected_result.capital_saved_estimate:.2f})',
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.success(f"Saved comparison: {output_path}")
    plt.close()


def main():
    """Generate visualizations from test results"""
    logger.info("Backtest visualization tool")
    logger.info("This will be called after running test_protected_system.py")
    logger.info("For now, this is a utility module to be imported")


if __name__ == '__main__':
    main()
