"""
Report Generator

Generate comprehensive HTML and text reports for strategy analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from loguru import logger

from .performance_metrics import PerformanceMetrics


class ReportGenerator:
    """
    Generate comprehensive strategy reports.

    Formats:
    - HTML (with CSS styling)
    - Text (terminal-friendly)
    - CSV (for further analysis)
    """

    def __init__(self):
        """Initialize report generator."""
        logger.info("Initialized report generator")

    def generate_html_report(
        self,
        metrics: Dict,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        strategy_name: str,
        output_path: Path,
    ) -> None:
        """
        Generate comprehensive HTML report.

        Args:
            metrics: Performance metrics dictionary
            equity_curve: Equity curve DataFrame
            trades: Trades DataFrame
            strategy_name: Name of strategy
            output_path: Output file path
        """
        html = []

        # Header
        html.append(self._html_header(strategy_name))

        # Executive Summary
        html.append(self._html_executive_summary(metrics))

        # Returns Section
        html.append(self._html_returns_section(metrics))

        # Risk Section
        html.append(self._html_risk_section(metrics))

        # Trading Statistics
        html.append(self._html_trading_section(metrics, trades))

        # Recent Trades
        html.append(self._html_recent_trades(trades))

        # Footer
        html.append(self._html_footer())

        # Write to file
        with open(output_path, 'w') as f:
            f.write("\n".join(html))

        logger.info(f"Generated HTML report: {output_path}")

    def _html_header(self, strategy_name: str) -> str:
        """Generate HTML header with CSS."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{strategy_name} - Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .metric-value.positive {{ color: #28a745; }}
        .metric-value.negative {{ color: #dc3545; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{strategy_name}</h1>
        <p>Comprehensive Performance Report</p>
    </div>
"""

    def _html_executive_summary(self, metrics: Dict) -> str:
        """Generate executive summary section."""
        return_class = "positive" if metrics.get('total_return', 0) > 0 else "negative"
        sharpe_class = "positive" if metrics.get('sharpe_ratio', 0) > 1 else ""

        return f"""
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {return_class}">
                    {metrics.get('total_return', 0)*100:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CAGR</div>
                <div class="metric-value">
                    {metrics.get('cagr', 0)*100:.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value {sharpe_class}">
                    {metrics.get('sharpe_ratio', 0):.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">
                    {metrics.get('max_drawdown', 0)*100:.2f}%
                </div>
            </div>
        </div>
    </div>
"""

    def _html_returns_section(self, metrics: Dict) -> str:
        """Generate returns analysis section."""
        return f"""
    <div class="section">
        <h2>Returns Analysis</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Return</td>
                <td>{metrics.get('total_return', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>CAGR</td>
                <td>{metrics.get('cagr', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Average Monthly Return</td>
                <td>{metrics.get('avg_monthly_return', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Best Month</td>
                <td>{metrics.get('best_month', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Worst Month</td>
                <td>{metrics.get('worst_month', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Positive Months</td>
                <td>{metrics.get('positive_months', 0)}/{metrics.get('total_months', 0)}</td>
            </tr>
        </table>
    </div>
"""

    def _html_risk_section(self, metrics: Dict) -> str:
        """Generate risk analysis section."""
        return f"""
    <div class="section">
        <h2>Risk Analysis</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Volatility (Annual)</td>
                <td>{metrics.get('volatility', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td>{metrics.get('max_drawdown', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Average Drawdown</td>
                <td>{metrics.get('avg_drawdown', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{metrics.get('sortino_ratio', 0):.2f}</td>
            </tr>
            <tr>
                <td>Calmar Ratio</td>
                <td>{metrics.get('calmar_ratio', 0):.2f}</td>
            </tr>
        </table>
    </div>
"""

    def _html_trading_section(self, metrics: Dict, trades: pd.DataFrame) -> str:
        """Generate trading statistics section."""
        return f"""
    <div class="section">
        <h2>Trading Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Trades</td>
                <td>{metrics.get('num_trades', 0)}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{metrics.get('win_rate', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td>{metrics.get('avg_win', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td>{metrics.get('avg_loss', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{metrics.get('profit_factor', 0):.2f}</td>
            </tr>
            <tr>
                <td>Expectancy</td>
                <td>{metrics.get('expectancy', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Largest Win</td>
                <td>{metrics.get('largest_win', 0)*100:.2f}%</td>
            </tr>
            <tr>
                <td>Largest Loss</td>
                <td>{metrics.get('largest_loss', 0)*100:.2f}%</td>
            </tr>
        </table>
    </div>
"""

    def _html_recent_trades(self, trades: pd.DataFrame, n: int = 10) -> str:
        """Generate recent trades table."""
        if len(trades) == 0:
            return ""

        recent = trades.tail(n)

        rows = []
        for _, trade in recent.iterrows():
            pnl_class = "positive" if trade['pnl_pct'] > 0 else "negative"
            rows.append(f"""
            <tr>
                <td>{trade.get('entry_time', 'N/A')}</td>
                <td>{trade.get('symbol', 'N/A')}</td>
                <td>{trade.get('direction', 'N/A')}</td>
                <td class="{pnl_class}">{trade['pnl_pct']*100:.2f}%</td>
                <td>{trade.get('duration', 0):.1f}h</td>
            </tr>
            """)

        return f"""
    <div class="section">
        <h2>Recent Trades (Last {min(n, len(trades))})</h2>
        <table>
            <tr>
                <th>Entry Time</th>
                <th>Symbol</th>
                <th>Direction</th>
                <th>P&L</th>
                <th>Duration</th>
            </tr>
            {"".join(rows)}
        </table>
    </div>
"""

    def _html_footer(self) -> str:
        """Generate HTML footer."""
        return """
    <div class="footer">
        <p>Generated by Expert Advisor Trading System</p>
        <p>Built with Claude Code</p>
    </div>
</body>
</html>
"""

    def __repr__(self) -> str:
        return "ReportGenerator()"
