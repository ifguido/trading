<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Binance-F0B90B?style=for-the-badge&logo=binance&logoColor=black" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
</p>

# CryptoTrader

**Automated cryptocurrency trading bot for Binance spot markets.** Event-driven architecture with AI-powered signal generation, multi-indicator technical analysis, market sentiment analysis, and real-time risk management.

Built entirely in async Python. Runs 24/7 on a $4 VPS.

**[See it live](http://165.232.168.225:8080/live)**

---

## Architecture

```
Market Data (WebSocket)
       |
OHLCVAggregator --> DataStore
       |                              MarketSentimentFeed
       |                              (funding rates + whale detection)
FeaturePipeline --> AI Model                    |
       |                                        |
Strategy (6-voter system) <--------------------+
       |
       | SignalEvent
       |
RiskManager --> validation + position sizing
       |
OrderExecutor --> Binance API
       |
FillHandler --> PortfolioTracker --> TrailingStopManager
       |
Web UI / Public Dashboard / Telegram / Database
```

All components communicate through an async **EventBus** (pub/sub). Zero tight coupling.

---

## Signal Generation: 6-Voter System

Every trading signal requires confluence from multiple independent indicators:

| # | Voter | Weight | What it measures |
|---|-------|--------|------------------|
| 1 | **MA Crossover** | 1.0 | Trend direction (fast vs slow moving average) |
| 2 | **RSI** | 1.0 | Overbought/oversold conditions |
| 3 | **MACD** | 1.0 | Momentum and trend strength |
| 4 | **Bollinger Bands** | 1.0 | Volatility and mean reversion |
| 5 | **AI Model** | 1.5 | GradientBoosting with 20+ engineered features |
| 6 | **Market Sentiment** | 1.0 | Funding rates (contrarian) + whale activity |

A signal is only emitted when confidence >= 0.4 (40% confluence). No single indicator can trigger a trade.

---

## Features

| Feature | Details |
|---|---|
| **Strategies** | Swing (6-voter confluence) and Scalping (order book imbalance) |
| **AI Model** | scikit-learn GradientBoosting trained on 365 days of OHLCV data |
| **Market Sentiment** | Funding rates from Binance Futures (contrarian signal) + whale trade detection ($50k+) |
| **Trailing Stop** | Dynamic exit — follows price up, sells on 3% retrace from peak. No fixed take-profit ceiling |
| **Risk Management** | Circuit breaker, position sizing, max drawdown, daily loss limit, mandatory stop-loss |
| **Web Dashboard** | Real-time portfolio, trades, signals. Protected by login with brute-force lockout |
| **Public Dashboard** | Read-only live view at `/live` — no login required. Link it anywhere |
| **Auto-Start** | Bot starts automatically on server boot with `AUTO_START=true` |
| **Paper Trading** | Full simulation mode without risking real funds |
| **Notifications** | Optional Telegram alerts on trades and system events |
| **Database** | SQLite (local) or PostgreSQL (production). Stores all trades, signals, and orders |
| **Deployment** | Docker + docker-compose. Single command to run |

---

## Risk Controls

```yaml
max_position_pct: 40%         # Max 40% of portfolio per trade
max_total_exposure_pct: 90%   # Max 90% total exposure
max_concurrent_positions: 6   # Max open positions at once
max_daily_loss_pct: 3%        # Circuit breaker: stops trading for the day
max_drawdown_pct: 10%         # Circuit breaker: stops trading on drawdown
mandatory_stop_loss: true     # Every order must have a stop-loss
trailing_stop: 3%             # Dynamic trailing stop (replaces fixed take-profit)
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

# Auto-start bot on server boot
AUTO_START=true
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

With `AUTO_START=true` in `.env`, the bot starts trading automatically when the container starts. No manual intervention needed — survives reboots.

---

## Project Structure

```
src/
  core/           Engine, EventBus, events, config loader
  strategy/       Swing (6-voter) and Scalping strategies
  ai/             Feature engineering, ML model, signal generation
  execution/      Order executor, paper executor, fill handler
  risk/           Risk manager, circuit breaker, position sizer, trailing stop, portfolio tracker
  data/           WebSocket feed, OHLCV aggregator, data store, market sentiment feed
  storage/        SQLAlchemy models, repository, DB init
  monitoring/     Health check, Telegram notifier
  web/            FastAPI app, auth, templates, static files, WebSocket, public dashboard
config/           YAML settings and strategy configs
scripts/          Entry points (run_web, run_bot, train_model)
models/           Trained ML models (joblib)
tests/            Unit and integration tests
```

---

## Web Authentication

The admin dashboard is protected with session-based authentication:

- Credentials configured via `WEB_USERNAME` and `WEB_PASSWORD_HASH` environment variables
- Password stored as SHA256 hash (never in plain text)
- **1 attempt per IP** — a single failed login locks the IP for 24 hours
- Session cookie with 7-day expiry

The public dashboard at `/live` requires no authentication.

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
  max_position_pct: 0.40
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
