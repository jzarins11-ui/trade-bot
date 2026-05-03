"""
db.py — Supabase database layer
Replaces local CSV/JSON files so bot.py (running on your PC)
and dashboard.py (running on Streamlit Cloud) share the same data.

All other files import from here. Nothing else needs to know about Supabase.
"""

import os
import json
import datetime
from datetime import timezone
import logging
from dotenv import load_dotenv

# Load environment variables from keys.env
load_dotenv("keys.env")

log = logging.getLogger(__name__)

# ── Try to import supabase ────────────────────────────────────────────────
try:
    from supabase import create_client, Client
    _supabase_available = True
except ImportError:
    _supabase_available = False
    log.warning("supabase-py not installed — falling back to local files.")

# ── Credentials (from environment or Streamlit secrets) ──────────────────
def _get_creds():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    # Try Streamlit secrets (works on Streamlit Cloud)
    if not url or not key:
        try:
            import streamlit as st
            if "SUPABASE_URL" in st.secrets:
                url = st.secrets["SUPABASE_URL"]
            if "SUPABASE_KEY" in st.secrets:
                key = st.secrets["SUPABASE_KEY"]
        except Exception:
            pass

    # Strip accidental trailing slashes that break the connection
    url = url.rstrip("/")

    return url, key

def _client():
    url, key = _get_creds()
    if not url or not key:
        return None
    if not _supabase_available:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        log.error(f"Supabase connect failed: {e}")
        return None

def _db_available() -> bool:
    url, key = _get_creds()
    return bool(url and key and _supabase_available)

# ─────────────────────────────────────────────────────────────────────────
# TRADES
# ─────────────────────────────────────────────────────────────────────────
def insert_trade(symbol: str, side: str, price: float, qty: int,
                 prob: float, pnl: float = None,
                 chaos_score: float = None, buy_threshold: float = None):
    """Write one trade row. Works with or without Supabase."""
    row = {
        "time":          datetime.datetime.now(timezone.utc).isoformat(),
        "symbol":        symbol,
        "side":          side,
        "price":         round(float(price), 4),
        "qty":           int(qty),
        "prob":          round(float(prob), 4) if prob is not None else None,
        "pnl":           round(float(pnl), 4) if pnl is not None else None,
        "chaos_score":   round(float(chaos_score), 4) if chaos_score is not None else None,
        "buy_threshold": round(float(buy_threshold), 4) if buy_threshold is not None else None,
    }

    if _db_available():
        try:
            _client().table("trades").insert(row).execute()
            return
        except Exception as e:
            log.error(f"Supabase insert_trade failed: {e} — writing locally.")

    # Fallback: local CSV
    import csv, os
    path = "trades.csv"
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_trades(limit: int = 200):
    """Load recent trades. Returns a list of dicts."""
    if _db_available():
        try:
            res = (_client().table("trades")
                   .select("*")
                   .order("time", desc=True)
                   .limit(limit)
                   .execute())
            return res.data or []
        except Exception as e:
            log.error(f"Supabase load_trades failed: {e}")

    # Fallback: local CSV
    try:
        import pandas as pd
        df = pd.read_csv("trades.csv")
        return df.tail(limit).to_dict("records")
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────
# METRICS  (live bot state — equity, chaos, RL, positions, etc.)
# ─────────────────────────────────────────────────────────────────────────
_METRICS_KEY = "live"

def save_metrics(data: dict):
    """Upsert the single live-metrics row."""
    payload = {
        "id":         _METRICS_KEY,
        "updated_at": datetime.datetime.now(timezone.utc).isoformat(),
        "data":       json.dumps(data),
    }

    if _db_available():
        try:
            _client().table("metrics").upsert(payload).execute()
            return
        except Exception as e:
            log.error(f"Supabase save_metrics failed: {e} — writing locally.")

    with open("metrics.json", "w") as f:
        data["updated_at"] = payload["updated_at"]
        json.dump(data, f, indent=2)


def load_metrics() -> dict:
    """Load the live metrics dict."""
    if _db_available():
        try:
            res = (_client().table("metrics")
                   .select("data")
                   .eq("id", _METRICS_KEY)
                   .execute())
            if res.data:
                return json.loads(res.data[0]["data"])
        except Exception as e:
            log.error(f"Supabase load_metrics failed: {e}")

    try:
        with open("metrics.json") as f:
            return json.load(f)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────
# RL STATE
# ─────────────────────────────────────────────────────────────────────────
_RL_KEY = "rl"

def save_rl_state(state: dict):
    payload = {
        "id":         _RL_KEY,
        "updated_at": datetime.datetime.now(timezone.utc).isoformat(),
        "data":       json.dumps(state),
    }

    if _db_available():
        try:
            _client().table("metrics").upsert(payload).execute()
            return
        except Exception as e:
            log.error(f"Supabase save_rl_state failed: {e}")

    with open("rl_state.json", "w") as f:
        json.dump(state, f, indent=2)


def load_rl_state() -> dict:
    if _db_available():
        try:
            res = (_client().table("metrics")
                   .select("data")
                   .eq("id", _RL_KEY)
                   .execute())
            if res.data:
                return json.loads(res.data[0]["data"])
        except Exception as e:
            log.error(f"Supabase load_rl_state failed: {e}")

    try:
        with open("rl_state.json") as f:
            return json.load(f)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────
# BACKTEST RESULTS
# ─────────────────────────────────────────────────────────────────────────
def save_backtest_results(results: list[dict]):
    if not results:
        return

    if _db_available():
        try:
            _client().table("backtest_results").insert(results).execute()
            return
        except Exception as e:
            log.error(f"Supabase save_backtest failed: {e} — writing locally.")

    import csv, os
    path = "backtest_results.csv"
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_header:
            w.writeheader()
        w.writerows(results)


def load_backtest_results() -> list[dict]:
    if _db_available():
        try:
            res = (_client().table("backtest_results")
                   .select("*")
                   .order("created_at", desc=True)
                   .limit(500)
                   .execute())
            return res.data or []
        except Exception as e:
            log.error(f"Supabase load_backtest failed: {e}")

    try:
        import pandas as pd
        df = pd.read_csv("backtest_results.csv")
        return df.to_dict("records")
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────
# KILL SWITCH  (dashboard writes, bot reads)
# ─────────────────────────────────────────────────────────────────────────
def send_kill_signal():
    """Called by dashboard Stop button."""
    payload = {"id": "kill", "updated_at": datetime.datetime.now(timezone.utc).isoformat(), "data": '"stop"'}
    if _db_available():
        try:
            _client().table("metrics").upsert(payload).execute()
            return True
        except Exception as e:
            log.error(f"Supabase kill signal failed: {e}")

    # Fallback: local file
    with open("STOP_BOT", "w") as f:
        f.write("stop")
    return True


def check_kill_signal() -> bool:
    """Called by bot every loop. Returns True if stop was requested."""
    if _db_available():
        try:
            res = (_client().table("metrics")
                   .select("data, updated_at")
                   .eq("id", "kill")
                   .execute())
            if res.data:
                value = json.loads(res.data[0]["data"])
                updated = res.data[0]["updated_at"]
                # Only act on kills from the last 2 minutes
                ts = datetime.datetime.fromisoformat(updated.replace("Z", ""))
                # If 'updated' is aware (has tzinfo), make sure we compare correctly
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                
                age = (datetime.datetime.now(timezone.utc) - ts).total_seconds()
                if value == "stop" and age < 120:
                    # Clear it so bot doesn't re-trigger on restart
                    _client().table("metrics").upsert(
                        {"id": "kill", "data": '"cleared"',
                         "updated_at": datetime.datetime.now(timezone.utc).isoformat()}
                    ).execute()
                    return True
        except Exception as e:
            log.error(f"Supabase kill check failed: {e}")

    # Fallback: local file
    if os.path.exists("STOP_BOT"):
        try:
            os.remove("STOP_BOT")
        except Exception:
            pass
        return True
    return False


def status() -> str:
    """Return 'supabase' or 'local' so dashboard can show connection status."""
    return "supabase" if _db_available() else "local"
