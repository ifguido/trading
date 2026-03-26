<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Binance-F0B90B?style=for-the-badge&logo=binance&logoColor=black" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
</p>

# CryptoTrader

**Automated cryptocurrency trading bot for Binance spot markets.** Event-driven architecture with AI-powered signal generation, multi-indicator technical analysis, real-time risk management, and a live web dashboard.

Built entirely in async Python. Designed to run 24/7 on a $4 VPS.

---

## Architecture

```
Market Data (WebSocket)
       |
OHLCVAggregator --> DataStore
       |
FeaturePipeline --> AI Model (GradientBoosting)
       |
Strategy (Swing / Scalping) --> SignalEvent
       |
RiskManager --> validation + position sizing
       |
OrderExecutor --> Binance API
       |
FillHandler --> PortfolioTracker
       |
Web UI / Telegram / Database
```

All components communicate through an async **EventBus** (pub/sub). Zero tight coupling.

---

## Features

| Feature | Details |
|---|---|
| **Strategies** | Swing (MA crossover + RSI + MACD + Bollinger) and Scalping (order book imbalance) |
| **AI Model** | scikit-learn GradientBoosting with 20+ engineered features. Acts as 5th voter in the strategy |
| **Risk Management** | Circuit breaker, position sizing, max drawdown, daily loss limit, mandatory stop-loss |
| **Web Dashboard** | Real-time prices, portfolio, trades, signals. Protected by login with brute-force lockout |
| **Paper Trading** | Full simulation mode without risking real funds |
| **Notifications** | Optional Telegram alerts on trades and system events |
| **Database** | SQLite (local) or PostgreSQL (production). Stores all trades, signals, and orders |
| **Deployment** | Docker + docker-compose. Single command to run |

---

## Risk Controls

```yaml
max_position_pct: 10%          # Max 10% of portfolio per trade
max_total_exposure_pct: 50%    # Max 50% total exposure
max_concurrent_positions: 6    # Max open positions at once
max_daily_loss_pct: 3%         # Circuit breaker: stops trading for the day
max_drawdown_pct: 10%          # Circuit breaker: stops trading on drawdown
mandatory_stop_loss: true      # Every order must have a stop-loss
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Binance account with API keys (spot trading enabled)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USER/cryptoTrader.git
cd cryptoTrader
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Web UI login
WEB_USERNAME=your_username
WEB_PASSWORD_HASH=your_sha256_hash
```

Generate your password hash:

```bash
python -c "import hashlib; print(hashlib.sha256(b'YOUR_PASSWORD').hexdigest())"
```

### 3. Configure trading pairs

Edit `config/settings.yaml` to set your pairs, timeframes, risk limits, and strategy parameters.

### 4. Train the AI model (optional)

```bash
python scripts/train_model.py
```

Downloads 365 days of OHLCV data from Binance and trains the GradientBoosting classifier. Models are saved to `models/`.

### 5. Run

```bash
python scripts/run_web.py
```

Open `http://localhost:8080`, log in, and hit **Start Bot**.

---

## Docker Deployment

```bash
docker-compose up -d --build
```

The bot runs on port `8080` with persistent volumes for data, config, and models.

```yaml
volumes:
  - ./data:/app/data        # SQLite database
  - ./config:/app/config    # Settings (editable)
  - ./models:/app/models    # AI models
  - ./.env:/app/.env        # Credentials
```

### Deploy to a VPS

```bash
# From your local machine
scp -r . root@YOUR_SERVER_IP:/opt/cryptotrader

# On the server
cd /opt/cryptotrader
docker-compose up -d --build

# Check logs
docker-compose logs -f
```

---

## Project Structure

```
src/
  core/           Engine, EventBus, events, config loader
  strategy/       Swing and Scalping strategies
  ai/             Feature engineering, ML model, signal generation
  execution/      Order executor, paper executor, fill handler
  risk/           Risk manager, circuit breaker, position sizer, portfolio tracker
  data/           WebSocket feed, OHLCV aggregator, data store
  storage/        SQLAlchemy models, repository, DB init
  monitoring/     Health check, Telegram notifier
  web/            FastAPI app, auth, templates, static files, WebSocket
config/           YAML settings and strategy configs
scripts/          Entry points (run_web, run_bot, train_model)
models/           Trained ML models (joblib)
tests/            Unit and integration tests
```

---

## Web Authentication

The dashboard is protected with session-based authentication:

- Credentials configured via `WEB_USERNAME` and `WEB_PASSWORD_HASH` environment variables
- Password stored as SHA256 hash (never in plain text)
- **1 attempt per IP** - a single failed login locks the IP for 24 hours
- Session cookie with 7-day expiry

---

## Configuration

All configuration lives in `config/settings.yaml`:

```yaml
exchange:
  name: binance
  mode: live              # live | paper | sandbox
  rate_limit: true

pairs:
  - symbol: BTC/USDC
    timeframes: [1m, 5m]
    strategy: swing

risk:
  max_position_pct: 0.10
  max_daily_loss_pct: 0.03
  mandatory_stop_loss: true

ai:
  enabled: true
  model_class: src.ai.local_model.LocalModel
  config:
    ai_weight: 1.5
    confidence_threshold: 0.4
```

---

## Tech Stack

- **Runtime**: Python 3.10+ (100% async)
- **Web**: FastAPI + Uvicorn + Jinja2
- **Exchange**: CCXT (Binance spot)
- **ML**: scikit-learn + pandas + numpy
- **Database**: SQLAlchemy 2.0 (SQLite / PostgreSQL)
- **Deployment**: Docker + docker-compose

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT
