# 🚀 HFT Bot v2.0 - Enterprise Ready High Frequency Trading System

![CI/CD](https://github.com/Alexandre Figura/hft-bot/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/codecov/c/github/Alexandre Figura/hft-bot)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/docker/image-size/Alexandre Figura/hft-bot)
[![Documentation](https://img.shields.io/readthedocs/hft-bot)](https://hft-bot.readthedocs.io)

A professional, production-ready HFT trading bot with enterprise features including ML predictions, advanced risk management, and real-time monitoring.

## ✨ Features

- **🏗️ Modular Architecture**: Clean, testable, and extensible design
- **📊 Advanced Analytics**: ML price prediction, market regime detection, iceberg order detection
- **🛡️ Risk Management**: Portfolio optimization, position sizing, drawdown control
- **📈 Real-time Monitoring**: Prometheus metrics, Grafana dashboards, multi-channel alerts
- **⚡ High Performance**: Numba-optimized indicators, async throughout, <10ms decision latency
- **🔧 Production Ready**: Docker support, CI/CD, comprehensive logging, state persistence

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Alexandre Figura/hft-bot.git
cd hft-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all features
pip install -e ".[dev,optimization,backtesting]"

# Setup pre-commit hooks
pre-commit install
```

### Basic Usage

```bash
# Initialize configuration
hft init

# Run paper trading
hft run --paper

# Run backtest
hft backtest data/BTCUSDT_1m.csv --start 2024-01-01 --end 2024-12-31

# Optimize parameters
hft optimize data/BTCUSDT_1m.csv --trials 1000 --metric sharpe

# Start with Docker
docker-compose up -d
```

## 📋 Configuration

Create `config/bot_config.yaml`:

```yaml
symbol: BTCUSDT
paper_trading: true
initial_capital: 10000

trading_params:
  decision_interval: 0.5
  min_signal_confluences: 2
  gross_take_profit: 0.0008
  gross_stop_loss: 0.0004
  max_position_size: 500
  kelly_fraction: 0.25

risk:
  max_drawdown: 0.15
  max_positions: 3
  max_exposure: 0.5
  daily_loss_limit: 0.05
```

## 📊 Monitoring

### Web Dashboard
Access the real-time dashboard at http://localhost:8080

### Prometheus Metrics
Metrics endpoint: http://localhost:8080/metrics

### Grafana Dashboards
```bash
# Start full monitoring stack
docker-compose --profile monitoring up -d
```
Access Grafana at http://localhost:3000 (admin/admin)

## 🧪 Development

### Running Tests
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_risk_manager.py -v

# Run with different Python versions
tox
```

### Code Quality
```bash
# Format code
black hft_bot/

# Lint
ruff check hft_bot/

# Type checking
mypy hft_bot/
```

### Building Documentation
```bash
# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

## 🏗️ Architecture

```
hft_bot/
├── core/          # Core interfaces and configuration
├── exchange/      # Exchange implementations (Binance, Paper)
├── analytics/     # Market analysis, ML, indicators
├── risk/          # Risk management and portfolio optimization
├── strategies/    # Trading strategies
├── infra/         # Infrastructure (logging, metrics, alerts)
└── cli/           # Command line interface
```

## 🚢 Deployment

### Docker
```bash
# Build image
docker build -t hft-bot:latest .

# Run container
docker run -d \
  --name hft-bot \
  --env-file .env \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/logs:/app/logs \
  -p 8080:8080 \
  hft-bot:latest
```

### Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -n hft-bot
```

## 📈 Performance

- **Latency**: <10ms decision time (p99)
- **Throughput**: 1000+ decisions/second
- **Memory**: <500MB typical usage
- **CPU**: Optimized with Numba JIT compilation

## ⚠️ Risk Warning

Trading cryptocurrency carries significant risk. This software is provided for educational purposes only. Never trade with money you cannot afford to lose. Past performance does not guarantee future results.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- 📖 [Documentation](https://hft-bot.readthedocs.io)
- 💬 [Discord Community](https://discord.gg/hft-bot)
- 🐛 [Issue Tracker](https://github.com/Alexandre Figura/hft-bot/issues)
- 📧 [Email Support](mailto:alexandre_figura@hotmail.com)

## 🙏 Acknowledgments

- Binance for the excellent API
- The Python async community
- All our contributors and users

---

Made with ❤️ by the HFT Bot Team
