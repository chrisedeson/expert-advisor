"""
Expert Advisor Trading System
A conservative Forex Expert Advisor with rigorous backtesting
"""

from setuptools import setup, find_packages

setup(
    name="expert-advisor",
    version="0.1.0",
    description="Forex Expert Advisor with comprehensive backtesting and optimization",
    author="Christopher Edeson",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ea-backtest=scripts.run_backtest:main",
            "ea-optimize=scripts.run_optimization:main",
            "ea-trade=scripts.trade_live:main",
        ]
    },
)
