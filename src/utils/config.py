"""
Configuration management for Expert Advisor
Loads and validates configuration from YAML files and environment variables
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager for the Expert Advisor system"""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.project_root = Path(__file__).parent.parent.parent

        # Load environment variables
        load_dotenv(self.project_root / ".env")

        # Load YAML configurations
        self.main_config = self._load_yaml("config.yaml")
        self.strategies_config = self._load_yaml("strategies.yaml")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_path = self.project_root / self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key

        Args:
            key: Configuration key in dot notation (e.g., 'trading.initial_capital')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.main_config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy"""
        return self.strategies_config.get(strategy_name, {})

    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable"""
        return os.getenv(key, default)

    # MT5 Configuration
    @property
    def mt5_login(self) -> int:
        """Get MT5 login from environment"""
        login = self.get_env("MT5_LOGIN")
        if not login:
            raise ValueError(
                "MT5_LOGIN not set in .env file. "
                "Copy .env.example to .env and add your MT5 credentials."
            )
        return int(login)

    @property
    def mt5_password(self) -> str:
        """Get MT5 password from environment"""
        password = self.get_env("MT5_PASSWORD")
        if not password or password == "your_password_here":
            raise ValueError(
                "MT5_PASSWORD not set in .env file. "
                "Copy .env.example to .env and add your MT5 credentials."
            )
        return password

    @property
    def mt5_server(self) -> str:
        """Get MT5 server from environment"""
        server = self.get_env("MT5_SERVER")
        if not server:
            raise ValueError(
                "MT5_SERVER not set in .env file. "
                "Copy .env.example to .env and add your MT5 credentials."
            )
        return server

    # Trading Configuration
    @property
    def initial_capital(self) -> float:
        """Get initial capital"""
        return float(self.get_env("INITIAL_CAPITAL", self.get("trading.initial_capital", 100)))

    @property
    def risk_per_trade(self) -> float:
        """Get risk per trade as decimal (e.g., 0.01 for 1%)"""
        return float(self.get_env("RISK_PER_TRADE", self.get("trading.risk_per_trade", 0.01)))

    @property
    def max_drawdown(self) -> float:
        """Get maximum allowed drawdown as decimal (e.g., 0.15 for 15%)"""
        return float(self.get_env("MAX_DRAWDOWN", self.get("trading.max_drawdown", 0.15)))

    # Data Configuration
    @property
    def data_dir(self) -> Path:
        """Get data directory path"""
        return self.project_root / "data"

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path"""
        cache_path = self.project_root / self.get("data.cache_dir", "data/cache")
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    @property
    def reports_dir(self) -> Path:
        """Get reports directory path"""
        reports_path = self.project_root / "reports"
        reports_path.mkdir(parents=True, exist_ok=True)
        return reports_path

    @property
    def logs_dir(self) -> Path:
        """Get logs directory path"""
        logs_path = self.project_root / "logs"
        logs_path.mkdir(parents=True, exist_ok=True)
        return logs_path

    # Currency Pairs
    @property
    def currency_pairs(self) -> list:
        """Get list of currency pairs to trade"""
        return self.get("pairs", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])

    # Logging Configuration
    @property
    def log_level(self) -> str:
        """Get log level"""
        return self.get_env("LOG_LEVEL", self.get("logging.level", "INFO"))

    def __repr__(self) -> str:
        return f"Config(server={self.mt5_server}, capital={self.initial_capital})"


# Global configuration instance
_config = None


def get_config() -> Config:
    """Get global configuration instance (singleton)"""
    global _config
    if _config is None:
        _config = Config()
    return _config
