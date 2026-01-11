"""
Transaction Cost Model

Realistic modeling of Forex trading costs:
- Spreads (bid-ask spread)
- Slippage (execution price deviation)
- Swap rates (overnight financing)
"""

from typing import Dict
from loguru import logger


class TransactionCostModel:
    """
    Model transaction costs for Forex trading.

    Costs included:
    - Spread: Bid-ask spread in pips
    - Slippage: Execution deviation (typically 50% of spread)
    - Swap: Overnight financing (simplified for now)
    """

    # Default costs for Exness Standard Account (in pips)
    DEFAULT_COSTS = {
        'EURUSD': {'spread': 0.7, 'slippage_factor': 0.5},
        'EURUSDm': {'spread': 0.7, 'slippage_factor': 0.5},
        'GBPUSD': {'spread': 0.9, 'slippage_factor': 0.5},
        'GBPUSDm': {'spread': 0.9, 'slippage_factor': 0.5},
        'USDJPY': {'spread': 0.8, 'slippage_factor': 0.5},
        'USDJPYm': {'spread': 0.8, 'slippage_factor': 0.5},
        'AUDUSD': {'spread': 0.9, 'slippage_factor': 0.5},
        'AUDUSDm': {'spread': 0.9, 'slippage_factor': 0.5},
    }

    # Pip values for different pairs (for standard lot)
    PIP_VALUES = {
        'EURUSD': 10.0,   # $10 per pip for 1 standard lot
        'EURUSDm': 10.0,
        'GBPUSD': 10.0,
        'GBPUSDm': 10.0,
        'USDJPY': 9.09,   # Approximate (varies with JPY rate)
        'USDJPYm': 9.09,
        'AUDUSD': 10.0,
        'AUDUSDm': 10.0,
    }

    def __init__(self, custom_costs: Dict = None, cost_multiplier: float = 1.0):
        """
        Initialize transaction cost model.

        Args:
            custom_costs: Override default costs (format: {symbol: {spread, slippage_factor}})
            cost_multiplier: Multiply all costs (for stress testing, e.g., 1.5 = +50% costs)
        """
        self.costs = self.DEFAULT_COSTS.copy()
        if custom_costs:
            self.costs.update(custom_costs)

        self.cost_multiplier = cost_multiplier

        logger.info(
            f"Initialized transaction cost model: "
            f"multiplier={cost_multiplier}x"
        )

    def get_spread(self, symbol: str) -> float:
        """
        Get spread in pips for a symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD" or "EURUSDm")

        Returns:
            Spread in pips
        """
        if symbol not in self.costs:
            logger.warning(f"Unknown symbol {symbol}, using default spread 1.0 pips")
            return 1.0 * self.cost_multiplier

        return self.costs[symbol]['spread'] * self.cost_multiplier

    def get_slippage(self, symbol: str) -> float:
        """
        Get estimated slippage in pips for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Slippage in pips (typically 50% of spread)
        """
        if symbol not in self.costs:
            logger.warning(f"Unknown symbol {symbol}, using default slippage 0.5 pips")
            return 0.5 * self.cost_multiplier

        spread = self.costs[symbol]['spread']
        slippage_factor = self.costs[symbol]['slippage_factor']
        return spread * slippage_factor * self.cost_multiplier

    def get_total_cost_pips(self, symbol: str) -> float:
        """
        Get total cost per round-turn trade in pips.

        Round-turn = entry + exit

        Args:
            symbol: Trading symbol

        Returns:
            Total cost in pips
        """
        spread = self.get_spread(symbol)
        slippage = self.get_slippage(symbol)

        # Round-turn cost = (spread + slippage) * 2 (entry + exit)
        return (spread + slippage) * 2

    def get_trade_cost(
        self, symbol: str, position_size: float = 1.0, price: float = 1.0
    ) -> float:
        """
        Get trade cost as a fraction of capital for position sizing.

        Args:
            symbol: Trading symbol
            position_size: Position size in lots (default: 1.0 = standard lot)
            price: Current price (for percentage calculation)

        Returns:
            Cost as fraction (e.g., 0.0001 = 0.01%)
        """
        cost_pips = self.get_total_cost_pips(symbol)
        pip_value = self.PIP_VALUES.get(symbol, 10.0)

        # Convert pips to price units
        if 'JPY' in symbol:
            # JPY pairs: 1 pip = 0.01
            cost_price = cost_pips * 0.01
        else:
            # Other pairs: 1 pip = 0.0001
            cost_price = cost_pips * 0.0001

        # Cost as fraction of position value
        cost_fraction = cost_price / price

        return cost_fraction

    def get_cost_in_dollars(
        self, symbol: str, position_size: float = 1.0
    ) -> float:
        """
        Get trade cost in dollars for a given position size.

        Args:
            symbol: Trading symbol
            position_size: Position size in lots

        Returns:
            Cost in USD
        """
        cost_pips = self.get_total_cost_pips(symbol)
        pip_value = self.PIP_VALUES.get(symbol, 10.0)

        # Cost = pips * pip_value * position_size
        return cost_pips * pip_value * position_size

    def apply_spread_to_data(self, data, symbol: str):
        """
        Apply spread to OHLC data (for more realistic backtesting).

        Modifies close prices to include spread:
        - Long entries: buy at ask (close + spread/2)
        - Short entries: sell at bid (close - spread/2)

        Args:
            data: DataFrame with OHLC data
            symbol: Trading symbol

        Returns:
            DataFrame with bid/ask prices
        """
        df = data.copy()
        spread_pips = self.get_spread(symbol)

        # Convert spread to price units
        if 'JPY' in symbol:
            spread_price = spread_pips * 0.01
        else:
            spread_price = spread_pips * 0.0001

        # Calculate bid/ask
        df['bid'] = df['close'] - spread_price / 2
        df['ask'] = df['close'] + spread_price / 2

        return df

    def __repr__(self) -> str:
        return (
            f"TransactionCostModel(multiplier={self.cost_multiplier}, "
            f"symbols={len(self.costs)})"
        )
