"""
dashboard.py — Trading bot monitoring dashboard
Reads all data from Supabase via db.py so it works on Streamlit Cloud
even when the bot is running on a different machine.
Run with: streamlit run dashboard.py
"""

import datetime
from datetime import timezone
import numpy as np
import pandas as pd
import streamlit as st

# db.py handles all Supabase reads (with local fallback)
import db

st.set_page_config(page_title="Trading Bot Monitor", page_icon="📈", layout="wide")

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="autorefresh")
except ImportError:
    pass

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def sharpe(returns):
    r = np.array(returns)
    if len(r) < 2 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))

def bot_is_running(metrics):
    t = metrics.get("updated_at", "")
    if not t: return False
    try:
        # Robust parsing for different formats
        t_clean = t.replace("Z", "").replace(" ", "T")
        if "." in t_clean:
            # Handle cases with more than 6 microseconds or different lengths
            parts = t_clean.split(".")
            t_clean = parts[0] + "." + parts[1][:6]
        
        updated = datetime.datetime.fromisoformat(t_clean)
        
        # Always compare in UTC — strip timezone info if present
        if updated.tzinfo is not None:
            updated = updated.replace(tzinfo=None)
        
        now_utc = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
        diff = (now_utc - updated).total_seconds()
        return diff < 600 # Reduced to 10 mins for better accuracy
    except Exception as e:
        return False

# ─────────────────────────────────────────────
# LOAD DATA  (all from db.py → Supabase or local)
# ─────────────────────────────────────────────
metrics   = db.load_metrics()
rl_state  = db.load_rl_state()
backtest  = db.load_backtest_results()
raw_trades = db.load_trades(limit=300)
trades    = pd.DataFrame(raw_trades) if raw_trades else None
storage   = db.status()   # 'supabase' or 'local'

# ─────────────────────────────────────────────
# HEADER + REMOTE CONTROL PANEL
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.title("📈 Trading Bot Monitor")
with col_h2:
    if st.button("🔄 Refresh"):
        st.rerun()

st.caption(f"Refreshed: {datetime.datetime.now().strftime('%H:%M:%S')}  |  "
           f"Storage: {'☁️ Supabase' if storage == 'supabase' else '💾 Local files'}")

st.subheader("🎛️ Remote Control")

running = bot_is_running(metrics)
paused  = metrics.get("market_status", "").startswith("PAUSED")

if storage != "supabase":
    st.warning("⚠️ Remote control requires Supabase. "
               "Add SUPABASE_URL and SUPABASE_KEY to your Streamlit secrets.")
else:
    # Status badge
    if running and not paused:
        st.success("🟢 Bot is RUNNING — trading normally")
    elif running and paused:
        st.warning("🟡 Bot is RUNNING — paused due to market chaos")
    else:
        st.error("🔴 Bot is STOPPED")

    st.write("")
    col_stop, col_pause, col_info = st.columns([1, 1, 2])

    with col_stop:
        if running:
            if st.button("⛔ Stop Bot", type="primary", use_container_width=True):
                db.send_kill_signal()
                st.warning("⛔ Stop signal sent. Bot shuts down within 60 seconds.")
        else:
            st.button("⛔ Stop Bot", disabled=True, use_container_width=True)
            st.caption("Bot is already stopped")

    with col_pause:
        currently_paused = metrics.get("manual_pause", False)
        if running:
            if currently_paused:
                if st.button("▶️ Resume Trading", use_container_width=True):
                    m = db.load_metrics()
                    m["manual_pause"] = False
                    db.save_metrics(m)
                    st.success("▶️ Resume signal sent.")
            else:
                if st.button("⏸️ Pause New Trades", use_container_width=True):
                    m = db.load_metrics()
                    m["manual_pause"] = True
                    db.save_metrics(m)
                    st.warning("⏸️ Paused. Bot keeps managing existing positions.")
        else:
            st.button("⏸️ Pause New Trades", disabled=True, use_container_width=True)

    with col_info:
        if not running:
            st.info("💡 **To start:** go to your PC and double-click **Start Trading Bot** "
                    "on the Desktop. Or run **INSTALL_SERVICE.bat** once to make the bot "
                    "start automatically every time Windows boots — no clicking needed.")
        elif paused:
            st.info("⏸️ Bot paused because market chaos > 0.6. Resumes automatically when calm.")
        else:
            st.info(f"✅ All good. Chaos score: {metrics.get('chaos_score', 0):.2f}/1.00")

st.divider()

# ─────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
equity = metrics.get("equity", 100_000)
c1.metric("Equity", f"${equity:,.2f}")
c2.metric("Open Positions", len(metrics.get("open_positions", {})))

if trades is not None and "side" in trades.columns:
    pnl_rows = trades[trades["side"] == "pnl_close"].copy()
    if not pnl_rows.empty:
        pnl_vals  = pnl_rows["pnl"].dropna().astype(float).values
        total_ret = float(np.prod(1 + pnl_vals) - 1) if len(pnl_vals) else 0.0
        win_rate  = float((pnl_vals > 0).mean()) if len(pnl_vals) else 0.0
        sh        = sharpe(pnl_vals)
        c3.metric("Total Return", f"{total_ret:.2%}")
        c4.metric("Win Rate",     f"{win_rate:.1%}")
        c5.metric("Sharpe Ratio", f"{sh:.2f}",
                  delta="⚠️ Low" if sh < 0.5 else None,
                  delta_color="inverse")
    else:
        for c, l in zip([c3, c4, c5], ["Total Return", "Win Rate", "Sharpe"]): 
            c.metric(l, "0.00%", help="No closed trades yet")
else:
    for c, l in zip([c3, c4, c5], ["Total Return", "Win Rate", "Sharpe"]): 
        c.metric(l, "0.00%", help="Waiting for trades...")

st.divider()

# ─────────────────────────────────────────────
# ① CHAOS METER
# ─────────────────────────────────────────────
st.subheader("① Market Chaos Meter")
chaos_score   = metrics.get("chaos_score", 0.0)
chaos_reason  = metrics.get("chaos_reason", "—")
market_status = metrics.get("market_status", "—")

col_chaos, col_info = st.columns([2, 3])
with col_chaos:
    if chaos_score < 0.3:   colour, label = "🟢", "CALM — trading normally"
    elif chaos_score < 0.6: colour, label = "🟡", "ELEVATED — bot is cautious"
    else:                   colour, label = "🔴", "CHAOTIC — new entries paused"
    st.metric("Chaos Score", f"{chaos_score:.2f} / 1.00", label)
    st.progress(min(float(chaos_score), 1.0))
    st.caption(f"{colour}  {market_status}")
with col_info:
    st.write("**Triggers chaos mode:**")
    st.write("- Volatility spike (2.5× above normal)")
    st.write("- All stocks crashing together (panic signal)")
    st.write("- Daily 5% loss limit hit")
    st.write(f"**Now:** `{chaos_reason}`")

st.divider()

# ─────────────────────────────────────────────
# ② REINFORCEMENT LEARNING
# ─────────────────────────────────────────────
st.subheader("② Bot Learning (Reinforcement Learning)")
rl = metrics.get("rl", {})

c1, c2, c3 = st.columns(3)
c1.metric("Buy Threshold",  f"{rl.get('buy_threshold',  0.65):.3f}",
          help="Started at 0.65. Drops after wins, rises after losses.")
c2.metric("Sell Threshold", f"{rl.get('sell_threshold', 0.35):.3f}")
total_reward = rl.get("total_reward", 0.0)
c3.metric("Cumulative Reward", f"{total_reward:+.2f}",
          delta="learning well" if total_reward > 0 else "still learning",
          delta_color="normal" if total_reward > 0 else "inverse")

if rl.get("win_rate") is not None:
    c1.metric("Recent Win Rate", f"{rl['win_rate']:.1%}")
if rl.get("avg_pnl") is not None:
    c2.metric("Avg Trade PnL", f"{rl['avg_pnl']:.2%}")
c3.metric("Trades Remembered", rl.get("trades", 0))

# RL threshold history chart
history = rl_state.get("trade_history", [])
if len(history) > 1:
    h_df = pd.DataFrame(history)
    if "buy_thresh_after" in h_df.columns:
        st.write("**Buy threshold over time** — dips = winning, spikes = losing:")
        st.line_chart(h_df[["buy_thresh_after"]].rename(columns={"buy_thresh_after": "Buy Threshold"}))

st.divider()

# ─────────────────────────────────────────────
# OPEN POSITIONS
# ─────────────────────────────────────────────
open_pos = metrics.get("open_positions", {})
if open_pos:
    st.subheader("Open Positions")
    st.dataframe(pd.DataFrame([
        {"Symbol": s, "Entry Price": f"${v['entry']:.2f}", "Qty": v["qty"]}
        for s, v in open_pos.items()
    ]), use_container_width=True, hide_index=True)
    st.divider()

# ─────────────────────────────────────────────
# TRADES + EQUITY CURVE
# ─────────────────────────────────────────────
if trades is not None and not trades.empty:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Recent Trades")
        display = trades[trades["side"].isin(["buy", "sell"])].tail(30)
        st.dataframe(display[::-1], use_container_width=True, hide_index=True)
    with col_r:
        st.subheader("Equity Curve")
        pnl_rows = trades[trades["side"] == "pnl_close"].copy()
        if not pnl_rows.empty and "time" in pnl_rows.columns:
            pnl_rows = pnl_rows.sort_values("time")
            pnl_rows["cumulative"] = (1 + pnl_rows["pnl"].astype(float)).cumprod() - 1
            st.line_chart(pnl_rows.set_index("time")["cumulative"])
        else:
            st.info("No closed trades yet — equity curve will appear here.")
    st.divider()
else:
    st.info("No trades yet — start the bot to see data here.")

# ─────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────
alerts = metrics.get("alerts", [])
if alerts:
    st.subheader("⚠️ Alerts")
    for a in reversed(alerts[-5:]):
        st.warning(f"**{a['subject']}** — {a['time'][:19]}\n\n{a['body']}")
    st.divider()

# ─────────────────────────────────────────────
# BACKTEST RESULTS
# ─────────────────────────────────────────────
if backtest:
    st.subheader("Walk-Forward Backtest Results")
    bt_df = pd.DataFrame(backtest)
    if not bt_df.empty and "symbol" in bt_df.columns:
        tab1, tab2 = st.tabs(["Summary", "All Folds"])
        with tab1:
            summary = bt_df.groupby("symbol").agg(
                Folds=("fold", "count"),
                Avg_Sharpe=("sharpe", "mean"),
                Avg_Return=("total_return", "mean"),
                Avg_Accuracy=("accuracy", "mean"),
            ).reset_index()
            summary["Avg_Return"]   = summary["Avg_Return"].apply(lambda x: f"{x:.2%}")
            summary["Avg_Accuracy"] = summary["Avg_Accuracy"].apply(lambda x: f"{x:.1%}")
            summary["Avg_Sharpe"]   = summary["Avg_Sharpe"].round(2)
            st.dataframe(summary, use_container_width=True, hide_index=True)
        with tab2:
            st.dataframe(bt_df, use_container_width=True, hide_index=True)
    st.divider()

# ─────────────────────────────────────────────
# PORTFOLIO WEIGHTS
# ─────────────────────────────────────────────
weights = metrics.get("portfolio_weights", {})
if weights:
    st.subheader("Portfolio Weights (Sharpe-Optimised)")
    w_df = pd.DataFrame([{"Symbol": s, "value": w} for s, w in weights.items()])
    st.bar_chart(w_df.set_index("Symbol")["value"])
    st.divider()

# ─────────────────────────────────────────────
# STOP GUIDE
# ─────────────────────────────────────────────
with st.expander("③ How to stop the bot"):
    st.write("1. **Dashboard** — click ⛔ Stop Bot above (works from anywhere via Supabase)")
    st.write("2. **Terminal** — type `stop` + Enter in the bot window")
    st.write("3. **Keyboard** — press Ctrl+C in the bot window")
    st.write("When stopped, bot asks if you want to close all open positions automatically.")

st.caption(f"Storage: **{storage}** | Bot log: `tail -f bot.log`")
