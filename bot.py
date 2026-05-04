"""
bot.py — Production trading bot
All data (trades, metrics, RL state, backtest results, kill switch)
now goes through db.py which writes to Supabase when configured,
or falls back to local files automatically.
"""

import os
import sys
import time
import json
import signal
import logging
import smtplib
import datetime
from datetime import timezone
import threading
from email.mime.text import MIMEText
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from polygon import RESTClient
    _polygon_available = True
except ImportError:
    _polygon_available = False

from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import minimize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import alpaca_trade_api as tradeapi

# ─────────────────────────────────────────────
# LOGGING (Moved to top to prevent NameErrors)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── db.py handles all storage (Supabase or local fallback) ───────────────
from dotenv import load_dotenv
load_dotenv("keys.env", override=True)  # Force use of keys.env file
import db

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SYMBOLS            = ["AAPL", "MSFT", "TSLA"]
CAPITAL            = 100_000
RISK_PER_TRADE     = 0.02  # Increased from 0.01
MAX_PORTFOLIO_RISK = 0.20  # Increased from 0.06 to allow more realistic exposure
MIN_SHARPE         = 0.5
DEGRADATION_WINDOW = 50

CHAOS_CORR_THRESHOLD = 0.85
CHAOS_VOL_MULTIPLIER = 2.5
CHAOS_DRAWDOWN_LIMIT = 0.05

RL_LEARNING_RATE       = 0.05
RL_BUY_THRESHOLD_INIT  = 0.40  # Lowered from 0.65 for testing
RL_SELL_THRESHOLD_INIT = 0.35
RL_MIN_BUY_THRESHOLD   = 0.40  # Lowered from 0.55
RL_MAX_BUY_THRESHOLD   = 0.80

# ── Load Keys ──────────────────────────────────
# (Already loaded in db.py, but safe to repeat or use defaults)
def _clean(val):
    if not val: return ""
    # Remove quotes, backticks, and ALL whitespace (including newlines/tabs)
    import re
    cleaned = str(val).strip()
    cleaned = re.sub(r'[\s\'"`]', '', cleaned)
    return cleaned

ALPACA_KEY    = _clean(os.environ.get("ALPACA_KEY", ""))
ALPACA_SECRET = _clean(os.environ.get("ALPACA_SECRET", ""))
ALPACA_URL    = _clean(os.environ.get("ALPACA_URL", ""))

# ── Robust Key Validation ──────────────────────
# Newer Alpaca keys can be 26 chars, secrets 44 chars. 
# We just clean up quotes/spaces and let the full key through.
if not ALPACA_KEY:
    log.error("CRITICAL: ALPACA_KEY is missing!")
if not ALPACA_SECRET:
    log.error("CRITICAL: ALPACA_SECRET is missing!")

# Debug (safe): Print first 4 chars of key to verify it's loaded
if ALPACA_KEY:
    log.info(f"Key loaded: {ALPACA_KEY[:4]}... (Length: {len(ALPACA_KEY)})")
if ALPACA_SECRET:
    log.info(f"Secret loaded: {ALPACA_SECRET[:4]}... (Length: {len(ALPACA_SECRET)})")

# ── Auto-Detect Paper vs Live ──────────────────
# If key starts with PK, it's PAPER. If AK, it's LIVE.
if ALPACA_KEY.startswith("PK"):
    log.info("Paper Keys detected. Forcing Paper URL.")
    ALPACA_URL = "https://paper-api.alpaca.markets"
elif ALPACA_KEY.startswith("AK"):
    log.info("Live Keys detected. Forcing Live URL.")
    ALPACA_URL = "https://api.alpaca.markets"
elif not ALPACA_URL:
    ALPACA_URL = "https://paper-api.alpaca.markets" # Default

# Strip trailing /v2 or slashes
ALPACA_URL = ALPACA_URL.rstrip("/").replace("/v2", "")

if not ALPACA_KEY or not ALPACA_SECRET:
    log.error("Live trading and clock checks will fail with 401 Unauthorized.")

POLYGON_KEY   = os.environ.get("POLYGON_KEY", "")
ALERT_EMAIL   = os.environ.get("ALERT_EMAIL", "")
SMTP_USER     = os.environ.get("SMTP_USER", "")
SMTP_PASS     = os.environ.get("SMTP_PASS", "")
SMTP_HOST     = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", 587))

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
# Create a function to get the API client to ensure it uses the latest cleaned keys
def get_alpaca_api():
    return tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version="v2")

api = get_alpaca_api()
polygon = RESTClient(POLYGON_KEY) if (_polygon_available and POLYGON_KEY) else None

# ─────────────────────────────────────────────
# KILL SWITCH
# Type 'stop' + Enter, press Ctrl+C,
# or click Stop in the dashboard (sends signal via Supabase)
# ─────────────────────────────────────────────
_shutdown = threading.Event()

def _keyboard_listener():
    while not _shutdown.is_set():
        try:
            if input().strip().lower() in ("stop", "quit", "exit", "q"):
                log.warning("Stop command received via keyboard.")
                _shutdown.set()
        except EOFError:
            break

def _signal_handler(sig, frame):
    log.warning("Ctrl+C — shutting down gracefully...")
    _shutdown.set()

def setup_kill_switch():
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if sys.stdin and sys.stdin.isatty():
        threading.Thread(target=_keyboard_listener, daemon=True).start()
        log.info("Kill switch active. Type 'stop' + Enter, Ctrl+C, "
                 "or use the Stop button in the dashboard.")
    else:
        log.info("Kill switch active via Signals or Dashboard (non-interactive).")

def should_stop() -> bool:
    if _shutdown.is_set():
        return True
    # Check Supabase (or local STOP_BOT file) for dashboard kill signal
    try:
        if db.check_kill_signal():
            log.warning("Kill signal received from dashboard.")
            _shutdown.set()
            return True
    except Exception:
        pass
    return False

def close_all_positions():
    log.warning("Closing all open positions...")
    try:
        api.close_all_positions()
        log.info("All positions closed.")
    except Exception as e:
        log.error(f"Failed to close positions: {e}")

# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
def fetch_polygon(symbol: str, days=30) -> pd.DataFrame:
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    aggs  = polygon.get_aggs(ticker=symbol, multiplier=5, timespan="minute",
                              from_=str(start), to=str(end), limit=50000, adjusted=True)
    rows = [{"Open": a.open, "High": a.high, "Low": a.low, "Close": a.close,
              "Volume": a.volume,
              "time": pd.Timestamp(a.timestamp, unit="ms", tz="UTC")} for a in aggs]
    return pd.DataFrame(rows).set_index("time").sort_index().dropna()

def fetch(symbol: str, period="30d") -> pd.DataFrame:
    """Fetch 5m data. Period defaults to 30d for better training."""
    if polygon:
        try:
            days = int(period.replace("d", "")) if "d" in period else 30
            return fetch_polygon(symbol, days=days)
        except Exception as e:
            log.warning(f"Polygon failed for {symbol}: {e}. Using yfinance.")
    
    # yfinance download
    df = yf.download(symbol, interval="5m", period=period, progress=False, auto_adjust=True)
    
    # Robust MultiIndex handling for yfinance 0.2+
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.levels[1]:
            df = df.xs(symbol, axis=1, level=1)
        else:
            df.columns = df.columns.get_level_values(0)
            
    # Ensure column names are standard
    rename_map = {col: col.capitalize() for col in df.columns if col.lower() in ["open", "high", "low", "close", "volume"]}
    if rename_map:
        df = df.rename(columns=rename_map)
        
    return df.dropna()

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
FEATURE_COLS = ["ema9", "ema21", "rsi", "Volume",
                "returns", "volatility", "momentum", "volume_spike",
                "bb_width", "atr"]

def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"]         = df["Close"].ewm(span=9).mean()
    df["ema21"]        = df["Close"].ewm(span=21).mean()
    df["rsi"]          = RSIIndicator(df["Close"]).rsi()
    df["returns"]      = df["Close"].pct_change()
    df["volatility"]   = df["returns"].rolling(10).std()
    df["momentum"]     = df["Close"] - df["Close"].shift(10)
    df["volume_spike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    sma20              = df["Close"].rolling(20).mean()
    df["bb_width"]     = (df["Close"].rolling(20).std() * 2) / sma20
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    df["atr"]          = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    df["target"]       = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()

# ─────────────────────────────────────────────
# CHAOS DETECTOR
# ─────────────────────────────────────────────
class ChaosDetector:
    def __init__(self):
        self.baseline_vol = {}

    def _vol(self, df):
        r = df["Close"].pct_change().dropna()
        return float(r.tail(20).std()) if len(r) >= 20 else 0.0

    def _update_baseline(self, symbol, vol):
        self.baseline_vol[symbol] = (
            vol if symbol not in self.baseline_vol
            else 0.95 * self.baseline_vol[symbol] + 0.05 * vol
        )

    def _vol_score(self, symbol, df):
        vol = self._vol(df)
        self._update_baseline(symbol, vol)
        base = self.baseline_vol.get(symbol, vol)
        if base == 0: return 0.0
        return min(1.0, max(0.0, (vol / base - 1.0) / (CHAOS_VOL_MULTIPLIER - 1.0)))

    def _corr_score(self, dfs):
        if len(dfs) < 2: return 0.0
        ret = pd.DataFrame({s: df["Close"].pct_change().dropna().tail(30)
                            for s, df in dfs.items()}).dropna()
        if len(ret) < 10: return 0.0
        m = ret.corr().values
        n = len(m)
        avg = float(np.mean(np.abs([m[i][j] for i in range(n) for j in range(n) if i != j])))
        return min(1.0, max(0.0, (avg - 0.5) / (CHAOS_CORR_THRESHOLD - 0.5)))

    def _dd_score(self):
        try:
            acc = api.get_account()
            eq, leq = float(acc.equity), float(acc.last_equity)
            if leq == 0: return 0.0
            dd = (leq - eq) / leq
            return 1.0 if dd > CHAOS_DRAWDOWN_LIMIT else round(dd / CHAOS_DRAWDOWN_LIMIT, 2)
        except Exception:
            return 0.0

    def score(self, dfs):
        vs = max((self._vol_score(s, df) for s, df in dfs.items()), default=0.0)
        cs = self._corr_score(dfs)
        ds = self._dd_score()
        final = vs * 0.40 + cs * 0.30 + ds * 0.30
        reasons = []
        if vs > 0.5:  reasons.append(f"volatility spike ({vs:.0%})")
        if cs > 0.5:  reasons.append(f"stocks in lockstep ({cs:.0%})")
        if ds >= 1.0: reasons.append("daily loss limit hit")
        return round(final, 3), (", ".join(reasons) or "normal")

chaos_detector = ChaosDetector()

# ─────────────────────────────────────────────
# REINFORCEMENT LEARNING
# ─────────────────────────────────────────────
class ThresholdAgent:
    def __init__(self):
        self.buy_threshold  = RL_BUY_THRESHOLD_INIT
        self.sell_threshold = RL_SELL_THRESHOLD_INIT
        self.trade_history  = []
        self.total_reward   = 0.0
        self._load()

    def _load(self):
        # Load from Supabase (or local file via db.py)
        state = db.load_rl_state()
        if state:
            self.buy_threshold  = state.get("buy_threshold",  RL_BUY_THRESHOLD_INIT)
            self.sell_threshold = state.get("sell_threshold", RL_SELL_THRESHOLD_INIT)
            self.total_reward   = state.get("total_reward",   0.0)
            self.trade_history  = state.get("trade_history",  [])[-200:]
            log.info(f"RL loaded: buy_thresh={self.buy_threshold:.3f} "
                     f"total_reward={self.total_reward:.2f} "
                     f"trades={len(self.trade_history)}")
        else:
            log.info("RL starting fresh.")

    def _save(self):
        # Save to Supabase (or local file via db.py)
        db.save_rl_state({
            "buy_threshold":  round(self.buy_threshold, 4),
            "sell_threshold": round(self.sell_threshold, 4),
            "total_reward":   round(self.total_reward, 4),
            "trade_history":  self.trade_history[-200:],
            "updated_at":     datetime.datetime.now(timezone.utc).isoformat(),
        })

    def learn(self, pnl: float, symbol: str, entry_prob: float):
        reward     = float(np.tanh(pnl * 20))
        adjustment = RL_LEARNING_RATE * reward
        old_buy    = self.buy_threshold

        self.buy_threshold = float(np.clip(
            self.buy_threshold - adjustment,
            RL_MIN_BUY_THRESHOLD, RL_MAX_BUY_THRESHOLD
        ))
        self.sell_threshold = float(np.clip(
            self.sell_threshold + adjustment,
            1.0 - RL_MAX_BUY_THRESHOLD, 1.0 - RL_MIN_BUY_THRESHOLD
        ))
        self.total_reward += reward
        self.trade_history.append({
            "time":              datetime.datetime.now(timezone.utc).isoformat(),
            "symbol":            symbol,
            "pnl":               round(pnl, 4),
            "reward":            round(reward, 4),
            "entry_prob":        round(entry_prob, 4),
            "buy_thresh_before": round(old_buy, 4),
            "buy_thresh_after":  round(self.buy_threshold, 4),
        })
        log.info(f"{'✅' if pnl > 0 else '❌'} RL [{symbol}]: "
                 f"pnl={pnl:+.2%} reward={reward:+.3f} "
                 f"buy_thresh {old_buy:.3f}→{self.buy_threshold:.3f}")
        self._save()

    def recent_performance(self, n=20):
        recent = self.trade_history[-n:]
        if not recent: return {}
        pnls = [r["pnl"] for r in recent]
        return {
            "trades":         len(recent),
            "win_rate":       round(sum(1 for p in pnls if p > 0) / len(pnls), 3),
            "avg_pnl":        round(float(np.mean(pnls)), 4),
            "total_reward":   round(self.total_reward, 3),
            "buy_threshold":  round(self.buy_threshold, 4),
            "sell_threshold": round(self.sell_threshold, 4),
        }

rl_agent = ThresholdAgent()

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
def train_rf(df):
    base  = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   min_samples_leaf=5, random_state=42)
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(df[FEATURE_COLS], df["target"])
    return model

def train_lstm(df):
    cols   = ["Close", "Volume", "ema9", "ema21", "rsi", "volatility", "atr"]
    scaler = MinMaxScaler()
    try:
        scaled = scaler.fit_transform(df[cols])
        X, y   = [], []
        for i in range(20, len(scaled) - 1):
            X.append(scaled[i - 20:i])
            y.append(df["target"].iloc[i])
        X, y = np.array(X), np.array(y)
        if len(X) < 50: return None, scaler
        
        # Build model
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2), LSTM(16), Dropout(0.2), Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        
        # Fit with limited epochs and small batch
        model.fit(X, y, epochs=10, batch_size=64, verbose=0,
                  validation_split=0.1, callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
        return model, scaler
    except Exception as e:
        log.warning(f"LSTM training failed: {e}. Falling back to RF only.")
        return None, scaler

def train_models(df):
    rf = train_rf(df)
    lstm, scaler = train_lstm(df)
    import gc
    gc.collect() # Force memory cleanup
    return rf, lstm, scaler

# ─────────────────────────────────────────────
# SIGNAL
# ─────────────────────────────────────────────
def get_signal(df, rf, lstm, scaler):
    if len(df) < 21: return "HOLD", 0.5
    rf_prob   = rf.predict_proba(df[FEATURE_COLS].iloc[-1:])[0][1]
    lstm_prob = 0.5
    if lstm is not None:
        try:
            cols   = ["Close", "Volume", "ema9", "ema21", "rsi", "volatility", "atr"]
            scaled = scaler.transform(df.iloc[-20:][cols])
            lstm_prob = float(lstm.predict(np.array([scaled]), verbose=0)[0][0])
        except Exception:
            pass
    prob = rf_prob * 0.55 + lstm_prob * 0.45
    if   prob > rl_agent.buy_threshold:  return "BUY",  prob
    elif prob < rl_agent.sell_threshold: return "SELL", prob
    else:                                return "HOLD", prob

# ─────────────────────────────────────────────
# WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────
def sharpe(returns, periods=252 * 78):
    r = np.array(returns)
    if len(r) < 2 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods))

def backtest_fold(df, train_idx, test_idx, symbol, fold):
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    rf, lstm, scaler = train_models(train)
    pnl_series, position, entry_price, correct = [], None, 0.0, 0
    for i in range(20, len(test)):
        sig, prob = get_signal(test.iloc[:i], rf, lstm, scaler)
        price     = float(test.iloc[i]["Close"])
        if sig != "HOLD":
            target_val = int(test.iloc[i]["target"])
            correct += (1 if sig == "BUY" else 0) == target_val
        if position is None and sig == "BUY":
            entry_price, position = price, "LONG"
        elif position == "LONG" and (sig == "SELL" or prob < rl_agent.sell_threshold):
            pnl_series.append((price - entry_price) / entry_price)
            position = None
    
    # Cleanup models to free memory
    del rf
    if lstm: del lstm
    import gc
    gc.collect()

    pnl_arr   = np.array(pnl_series) if pnl_series else np.array([0.0])
    total_ret = float(np.prod(1 + pnl_arr) - 1)
    sh        = sharpe(pnl_arr)
    acc       = correct / max(1, len(pnl_series))
    log.info(f"Backtest {symbol} fold {fold}: Sharpe={sh:.2f} Return={total_ret:.2%} Acc={acc:.2%}")
    return {"symbol": symbol, "fold": fold,
            "train_start": str(train.index[0]), "train_end": str(train.index[-1]),
            "test_start": str(test.index[0]),  "test_end":  str(test.index[-1]),
            "accuracy": round(acc, 4), "sharpe": round(sh, 4), "total_return": round(total_ret, 4)}

def walk_forward_backtest(symbol, n_splits=5):
    log.info(f"Backtesting {symbol}...")
    df      = features(fetch(symbol))
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        if len(train_idx) < 100 or len(test_idx) < 20: continue
        results.append(backtest_fold(df, train_idx, test_idx, symbol, fold))
    if results:
        db.save_backtest_results(results)   # → Supabase or local CSV
    return results

# ─────────────────────────────────────────────
# PORTFOLIO OPTIMIZER
# ─────────────────────────────────────────────
def optimize_portfolio(returns_dict):
    symbols = list(returns_dict.keys())
    n       = len(symbols)
    min_len = min(len(v) for v in returns_dict.values())
    R       = np.column_stack([returns_dict[s][-min_len:] for s in symbols])
    mu, cov = R.mean(axis=0), (np.cov(R, rowvar=False) if min_len > 1 else np.eye(n))
    result  = minimize(
        lambda w: -(w @ mu) / np.sqrt(w @ cov @ w + 1e-8),
        np.ones(n) / n, method="SLSQP",
        bounds=[(0.0, 0.5)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1}
    )
    weights = result.x if result.success else np.ones(n) / n
    return {s: float(w) for s, w in zip(symbols, weights)}

# ─────────────────────────────────────────────
# MARKET HOURS
# ─────────────────────────────────────────────
def is_market_open():
    """
    Checks if the US market is open. 
    Uses Alpaca clock first, falls back to manual time check if API fails or seems wrong.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    # May is Daylight Savings (EDT = UTC-4)
    # Market: 9:30 AM - 4:00 PM ET -> 13:30 - 20:00 UTC
    hour_utc = now_utc.hour + now_utc.minute / 60.0
    is_weekend = now_utc.weekday() >= 5
    likely_open = (not is_weekend) and (13.5 <= hour_utc <= 20.0)

    try:
        clock = api.get_clock()
        if clock.is_open:
            return True
        
        if likely_open:
            log.warning(f"Alpaca says CLOSED but UTC time {hour_utc:.2f} suggests OPEN. Alpaca next_open: {clock.next_open}")
            # If it's a weekday and during hours, Alpaca might be wrong or we have a key issue
            # We'll trust the time check as a backup if Alpaca is being strange
            return True
            
        return False
    except Exception as e:
        log.error(f"Alpaca clock error: {e}")
        if likely_open:
            log.info(f"Manual fallback: Market should be OPEN (UTC {hour_utc:.2f}).")
            return True
        return False

# ─────────────────────────────────────────────
# POSITION MANAGER
# ─────────────────────────────────────────────
class Position:
    def __init__(self, symbol, price, qty, entry_prob, atr=None):
        self.symbol, self.entry, self.qty = symbol, price, qty
        self.sold_qty  = 0
        # Use 2.5 * ATR for trailing stop if available, else 1.5% fixed
        self.atr_mult = 2.5
        self.stop_dist = (atr * self.atr_mult) if atr else (price * 0.015)
        self.trailing  = price - self.stop_dist
        self.tp        = price + (self.stop_dist * 1.5) # Initial TP at 1.5x risk
        self.entry_prob = entry_prob
        self.entry_time = datetime.datetime.now(timezone.utc)

    @property
    def remaining_qty(self): return self.qty - self.sold_qty

    def update(self, price, current_atr=None):
        qty_to_sell = 0
        
        # ── Time-Based Exit (Optional) ──
        # If position is held too long without profit, exit
        age_hours = (datetime.datetime.now(timezone.utc) - self.entry_time).total_seconds() / 3600
        if age_hours > 48 and price < self.entry * 1.005:
            log.info(f"Time-based exit for {self.symbol} (held {age_hours:.1f}h)")
            return True, self.remaining_qty

        # Move trailing stop up
        if current_atr:
            self.stop_dist = current_atr * self.atr_mult
        
        new_trailing = price - self.stop_dist
        if new_trailing > self.trailing:
            self.trailing = new_trailing

        # Partial Take Profit
        if price >= self.tp and self.remaining_qty > 0:
            partial        = max(1, self.remaining_qty // 2)
            self.sold_qty += partial
            self.tp       += self.stop_dist # Move TP up by one risk unit
            qty_to_sell    = partial
            log.info(f"Partial profit {self.symbol}: selling {partial} shares @ {price:.2f}")

        if price <= self.trailing:
            return True, self.remaining_qty
        return False, qty_to_sell

# ─────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────
class AlertSystem:
    def __init__(self):
        self.recent_pnl     = deque(maxlen=DEGRADATION_WINDOW)
        self.alert_cooldown = {}

    def record_pnl(self, pnl):
        self.recent_pnl.append(pnl)
        if len(self.recent_pnl) >= 20:
            sh = sharpe(list(self.recent_pnl), periods=252)
            if sh < MIN_SHARPE:
                self._send_alert("Strategy Degradation",
                                 f"Rolling Sharpe={sh:.2f} (min={MIN_SHARPE}). "
                                 f"Buy threshold: {rl_agent.buy_threshold:.3f}")

    def _send_alert(self, subject, body):
        now = datetime.datetime.now(timezone.utc)
        if subject in self.alert_cooldown:
            if (now - self.alert_cooldown[subject]).seconds < 3600: return
        self.alert_cooldown[subject] = now
        log.warning(f"ALERT: {subject} — {body}")
        # Append alert into the live metrics so dashboard shows it
        m = db.load_metrics()
        m.setdefault("alerts", []).append({"time": now.isoformat(), "subject": subject, "body": body})
        db.save_metrics(m)
        # Email
        if ALERT_EMAIL and SMTP_USER:
            try:
                msg = MIMEText(body)
                msg["Subject"] = f"[TradingBot] {subject}"
                msg["From"]    = SMTP_USER
                msg["To"]      = ALERT_EMAIL
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                    s.starttls(); s.login(SMTP_USER, SMTP_PASS)
                    s.sendmail(SMTP_USER, ALERT_EMAIL, msg.as_string())
            except Exception as e:
                log.error(f"Email failed: {e}")

alert_system = AlertSystem()

# ─────────────────────────────────────────────
# ORDER EXECUTION
# ─────────────────────────────────────────────
def get_equity():
    try:    return float(api.get_account().equity)
    except: return CAPITAL

def submit_order(symbol, side, qty, prob, price):
    if qty <= 0: return 0.0
    try:
        global api
        api.submit_order(symbol=symbol, qty=qty, side=side,
                         type="market", time_in_force="day")
        log.info(f"ORDER {side.upper()} {qty}x {symbol} @ ~{price:.2f} prob={prob:.3f}")
        # Write trade to Supabase (or local CSV via db.py)
        db.insert_trade(symbol=symbol, side=side, price=price, qty=qty,
                        prob=prob, buy_threshold=rl_agent.buy_threshold)
        return price
    except Exception as e:
        # If unauthorized, try one refresh of the API client
        if "unauthorized" in str(e).lower():
            try:
                log.warning(f"Unauthorized error for {symbol}. Refreshing API client and retrying...")
                api = get_alpaca_api()
                api.submit_order(symbol=symbol, qty=qty, side=side,
                                 type="market", time_in_force="day")
                log.info(f"ORDER {side.upper()} {qty}x {symbol} success after refresh")
                return price
            except Exception as e2:
                log.error(f"Order failed {side} {symbol} after refresh: {e2}")
        else:
            log.error(f"Order failed {side} {symbol}: {e}")
        return 0.0

def record_close(symbol, entry, exit_price, qty, entry_prob, chaos_score):
    pnl = (exit_price - entry) / entry if entry else 0.0
    alert_system.record_pnl(pnl)
    rl_agent.learn(pnl, symbol, entry_prob)
    # Write closed-trade PnL to Supabase (or local CSV via db.py)
    db.insert_trade(symbol=symbol, side="pnl_close", price=exit_price, qty=qty,
                    prob=None, pnl=pnl, chaos_score=chaos_score,
                    buy_threshold=rl_agent.buy_threshold)

def size(price, prob, weight=1.0):
    risk = get_equity() * RISK_PER_TRADE * (prob * 2) * weight
    return max(1, int(risk / price))

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
portfolio = {}

def run():
    setup_kill_switch()
    log.info(f"Storage mode: {db.status()}")  # logs 'supabase' or 'local'

    # ── FAST START CHECK ──
    # Check if we already have backtest results in Supabase/Local.
    # If we do, we can skip the slow backtesting phase to get to live trading faster.
    existing_bt = db.load_backtest_results()
    if existing_bt:
        log.info("Existing backtest results found. Skipping slow backtesting phase.")
    else:
        log.info("=== Walk-forward backtests ===")
        for s in SYMBOLS:
            if should_stop(): break
            try:    walk_forward_backtest(s)
            except Exception as e: log.warning(f"Backtest failed {s}: {e}")

    log.info("=== Training live models ===")
    models = {}
    for s in SYMBOLS:
        if should_stop(): break
        try:
            # Fetch 60 days for initial training to get a robust model
            df = features(fetch(s, period="60d"))
            models[s] = train_models(df)
            log.info(f"Models ready: {s} ({len(df)} bars)")
        except Exception as e:
            log.error(f"Model training failed {s}: {e}")

    symbol_returns = {s: [] for s in SYMBOLS}
    weights        = {s: 1.0 / len(SYMBOLS) for s in SYMBOLS}
    chaos_score, chaos_reason = 0.0, "starting"

    log.info("=== Live trading loop ===")

    while not should_stop():

        if not is_market_open():
            log.info("Market closed — sleeping 5 min")
            # Still write status so dashboard shows bot is running
            try:
                db.save_metrics({
                    "open_positions":    {s: {"entry": p.entry, "qty": p.remaining_qty}
                                          for s, p in portfolio.items()},
                    "equity":            get_equity(),
                    "chaos_score":       0.0,
                    "chaos_reason":      "market closed",
                    "market_status":     "MARKET CLOSED — waiting for open",
                    "rl":                rl_agent.recent_performance(),
                    "portfolio_weights": weights,
                    "alerts":            db.load_metrics().get("alerts", []),
                })
            except Exception as e:
                log.error(f"Metrics write failed: {e}")
            for _ in range(300):
                if should_stop(): break
                time.sleep(1)
            continue

        # Fetch all bars
        all_dfs = {}
        for s in SYMBOLS:
            try:    all_dfs[s] = features(fetch(s))
            except Exception as e: log.error(f"Fetch failed {s}: {e}")

        # Chaos check
        chaos_score, chaos_reason = chaos_detector.score(all_dfs)
        
        # ── Market Sentiment Filter (Extra Protection) ──
        # If the majority of symbols are trending down, be extra cautious
        bearish_count = 0
        for s, df in all_dfs.items():
            if len(df) > 20:
                short_ema = df["Close"].ewm(span=9).mean().iloc[-1]
                long_ema  = df["Close"].ewm(span=21).mean().iloc[-1]
                if short_ema < long_ema: bearish_count += 1
        
        if bearish_count >= len(SYMBOLS) * 0.6:
            chaos_score = max(chaos_score, 0.4)
            chaos_reason += " (broad bearish trend)"

        chaotic = chaos_score > 0.6
        if chaotic:
            log.warning(f"CHAOS ({chaos_score:.2f}): {chaos_reason} — no new entries")
        else:
            log.info(f"Market calm (chaos={chaos_score:.2f})")

        # Manual pause check (set from dashboard Pause button via Supabase)
        manual_pause = db.load_metrics().get("manual_pause", False)
        if manual_pause:
            log.info("Manual pause active — skipping new entries this cycle")
            chaotic = True  # reuse the same gate that blocks new entries

        # Refresh portfolio weights every 50 closed trades
        total_closed = sum(len(v) for v in symbol_returns.values())
        if total_closed > 0 and total_closed % 50 == 0:
            arrays = {s: np.array(v) for s, v in symbol_returns.items() if len(v) > 5}
            if len(arrays) == len(SYMBOLS):
                weights = optimize_portfolio(arrays)
                log.info(f"Weights: {weights}")

        for s in SYMBOLS:
            if should_stop(): break
            if s not in models or s not in all_dfs: continue
            try:
                df = all_dfs[s]
                sig, prob = get_signal(df, *models[s])
                price     = float(df.iloc[-1]["Close"])
                atr       = float(df.iloc[-1]["atr"]) if "atr" in df.columns else None
                log.info(f"{s} {sig} prob={prob:.3f} thresh={rl_agent.buy_threshold:.3f} ${price:.2f}")

                if s not in portfolio and sig == "BUY" and not chaotic:
                    exposure = sum(p.entry * p.remaining_qty for p in portfolio.values())
                    if exposure / max(get_equity(), 1) > MAX_PORTFOLIO_RISK:
                        log.info(f"Skipping {s} — risk cap")
                        continue
                    qty = size(price, prob, weight=weights.get(s, 1 / len(SYMBOLS)))
                    if submit_order(s, "buy", qty, prob, price):
                        portfolio[s] = Position(s, price, qty, entry_prob=prob, atr=atr)

                elif s in portfolio:
                    pos = portfolio[s]
                    should_exit, sell_qty = pos.update(price, current_atr=atr)
                    if sell_qty > 0 and not should_exit:
                        submit_order(s, "sell", sell_qty, prob, price)
                    if should_exit or sig == "SELL":
                        if pos.remaining_qty > 0:
                            submit_order(s, "sell", pos.remaining_qty, prob, price)
                        record_close(s, pos.entry, price, pos.qty, pos.entry_prob, chaos_score)
                        symbol_returns[s].append((price - pos.entry) / pos.entry)
                        del portfolio[s]

            except Exception as e:
                log.error(f"Error {s}: {e}", exc_info=True)

        # Push live metrics to Supabase so dashboard can read them
        db.save_metrics({
            "open_positions":    {s: {"entry": p.entry, "qty": p.remaining_qty}
                                  for s, p in portfolio.items()},
            "equity":            get_equity(),
            "chaos_score":       chaos_score,
            "chaos_reason":      chaos_reason,
            "market_status":     "PAUSED (chaotic)" if chaotic else "ACTIVE",
            "rl":                rl_agent.recent_performance(),
            "portfolio_weights": weights,
            "alerts":            db.load_metrics().get("alerts", []),
        })

        for _ in range(60):
            if should_stop(): break
            time.sleep(1)

    # Graceful shutdown
    log.warning("Shutting down...")
    if portfolio:
        log.warning(f"Open positions: {list(portfolio.keys())}")
        ans = "n"
        if sys.stdin and sys.stdin.isatty():
            try:    ans = input("Close all open positions? (y/n): ").strip().lower()
            except: ans = "n"
        
        if ans == "y": 
            close_all_positions()
        else: 
            log.warning("Positions left open — check Alpaca.")
    log.info("Bot stopped. Goodbye.")

if __name__ == "__main__":
    run()
