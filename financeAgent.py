import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import yfinance as yf
import os
import json
from dotenv import load_dotenv

# ─── Environment ───────────────────────────────────────────────
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# ─── Watchlist persistence ─────────────────────────────────────
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlist, f, indent=2)

# ─── SAFE Yahoo Finance Fetch (RATE-LIMIT FIX) ─────────────────
@st.cache_data(ttl=300)  # cache for 5 minutes
def get_stock_info(symbol):
    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="AI Equity Research Agent", layout="wide")
st.title("🧠 AI Equity Research Agent")

# ─── Session State ─────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None

if "analyze_trigger" not in st.session_state:
    st.session_state.analyze_trigger = 0

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📌 Watchlist")

    new_sym = st.text_input(
        "Add Symbol",
        placeholder="TCS, AAPL, RELIANCE",
        key="add_input"
    )

    if st.button("Add to Watchlist", use_container_width=True):
        sym = new_sym.strip().upper()
        if sym and sym not in st.session_state.watchlist:
            st.session_state.watchlist.append(sym)
            save_watchlist(st.session_state.watchlist)
            st.rerun()

    st.divider()

    if st.session_state.watchlist:
        for idx, sym in enumerate(st.session_state.watchlist):
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(sym, key=f"sel_{sym}_{idx}", use_container_width=True):
                    st.session_state.selected_symbol = sym
                    st.session_state.analyze_trigger += 1
                    st.rerun()
            with col2:
                if st.button("×", key=f"rm_{sym}_{idx}"):
                    st.session_state.watchlist.pop(idx)
                    save_watchlist(st.session_state.watchlist)
                    if st.session_state.selected_symbol == sym:
                        st.session_state.selected_symbol = None
                    st.rerun()
    else:
        st.caption("Watchlist is empty")

    st.divider()
    st.header("Quick Position Calc")

    qty = st.number_input("Quantity", min_value=0, step=1, value=0)
    buy_price = st.number_input("Avg Buy Price", min_value=0.0, value=0.0, step=0.01)

# ─── Main Area ─────────────────────────────────────────────────
ticker_input = st.text_input(
    "Stock Symbol",
    value=st.session_state.selected_symbol or "TCS",
    key=f"main_ticker_{st.session_state.analyze_trigger}"
)

should_analyze = (
    st.button("Analyze Stock", type="primary", use_container_width=True)
    or st.session_state.analyze_trigger > 0
)

symbol_to_use = ticker_input.strip().upper() if ticker_input else st.session_state.selected_symbol

# ─── Helpers ───────────────────────────────────────────────────
def safe_format(value, fmt="{:,.2f}", fallback="—"):
    if value is None:
        return fallback
    try:
        return fmt.format(value)
    except:
        return str(value)

# ─── Run Analysis ──────────────────────────────────────────────
if should_analyze and symbol_to_use:

    st.session_state.analyze_trigger = 0

    symbol = symbol_to_use
    if "." not in symbol:
        symbol += ".NS"

    stock = yf.Ticker(symbol)

    # ✅ SAFE info fetch (rate-limit protected)
    info = get_stock_info(symbol)

    # robust fetching
    try:
        fast = stock.fast_info
        current_price = fast.get("lastPrice") or fast.get("regularMarketPreviousClose")
        market_cap = fast.get("marketCap")
    except:
        current_price = None
        market_cap = None

    current_price = current_price or info.get("currentPrice")
    market_cap = market_cap or info.get("marketCap")
    trailing_pe = info.get("trailingPE") or info.get("forwardPE")
    roe = info.get("returnOnEquity")
    pb_ratio = info.get("priceToBook")
    eps = info.get("trailingEps")
    dividend_yield = info.get("dividendYield")
    debt_to_equity = info.get("debtToEquity")

    tabs = st.tabs([
        "Overview",
        "Investment Thesis",
        "News & Events",
        "Portfolio Advice"
    ])

    # ─── Overview ──────────────────────────────────────────────
    with tabs[0]:
        st.subheader(f"📊 {symbol}")

        st.markdown(f"**Company:** {info.get('longName', symbol)}")
        st.markdown(f"**Sector / Industry:** {info.get('sector', '—')} / {info.get('industry', '—')}")

        with st.expander("Business Summary"):
            st.caption(info.get("longBusinessSummary", "No description available."))

        cols = st.columns(4)
        cols[0].metric("Current Price", safe_format(current_price))
        cols[1].metric("Market Cap", safe_format(market_cap, "{:,.0f}"))
        cols[2].metric("P/E Ratio", safe_format(trailing_pe, "{:.1f}"))
        cols[3].metric("ROE", safe_format(roe, "{:.1%}"))

        st.markdown("---")
        cols2 = st.columns(4)
        cols2[0].metric("Price / Book", safe_format(pb_ratio, "{:.2f}"))
        cols2[1].metric("EPS (TTM)", safe_format(eps, "{:.2f}"))
        cols2[2].metric("Dividend Yield", safe_format(dividend_yield, "{:.2%}"))
        cols2[3].metric("Debt / Equity", safe_format(debt_to_equity, "{:.1f}"))

    # ─── Agents (unchanged) ────────────────────────────────────
    @st.cache_resource
    def get_agents():
        fundamentals = Agent(
            name="Fundamentals",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[YFinanceTools(stock_fundamentals=True)],
            instructions=(
                "You are a financial analyst. "
                "Use ONLY provided yfinance data. "
                "Structure output: Valuation | Profitability | Growth | Leverage | Red Flags."
            ),
            markdown=True,
        )

        news = Agent(
            name="News",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[DuckDuckGo()],
            instructions=(
                "Find 3–5 most important company events in last 9 months. "
                "Format as numbered list with date, event, impact."
            ),
            markdown=True,
        )

        decision = Agent(
            name="Decision",
            model=Groq(id="llama-3.3-70b-versatile"),
            instructions=(
                "You are a senior equity analyst. "
                "Use ONLY provided fundamentals and news. "
                "DO NOT invent numbers. "
                "Return exactly:\n\n"
                "## Verdict\n"
                "## Confidence (High/Medium/Low)\n"
                "## Time Horizon\n"
                "## Key Reasons (3–5 bullets)\n"
                "## Main Risks (2–4 bullets)"
            ),
            markdown=True,
        )
        return fundamentals, news, decision

    fundamental_agent, news_agent, decision_agent = get_agents()

    # ─── Investment Thesis ─────────────────────────────────────
    with tabs[1]:
        with st.spinner("Building investment thesis..."):
            fund_md = fundamental_agent.run(
                f"Analyze fundamentals of {symbol}"
            ).content

            thesis = decision_agent.run(
                f"Fundamentals:\n{fund_md}\n\nProvide investment thesis for {symbol}"
            ).content

        st.markdown(thesis)

        with st.expander("Raw Fundamentals Output"):
            st.markdown(fund_md)

    # ─── News & Events ─────────────────────────────────────────
    with tabs[2]:
        with st.spinner("Fetching recent catalysts..."):
            events_md = news_agent.run(
                f"Important news and events for {symbol}"
            ).content

        st.markdown("### 🛎️ Recent News & Catalysts")
        st.markdown(events_md)

    # ─── Portfolio Advice ──────────────────────────────────────
    with tabs[3]:
        if qty > 0 and buy_price > 0 and current_price:
            pnl = (current_price - buy_price) * qty
            pnl_pct = ((current_price - buy_price) / buy_price) * 100

            st.metric(
                "Unrealized P&L",
                safe_format(pnl, "{:,.0f}"),
                f"{pnl_pct:+.1f}%",
                delta_color="normal" if pnl >= 0 else "inverse"
            )

            with st.spinner("Generating position advice..."):
                advice = decision_agent.run(
                    f"User holds {qty} shares of {symbol} at {buy_price}. "
                    f"Current price is {current_price}. "
                    "Should they HOLD, ADD, TRIM, or EXIT?"
                ).content

            st.markdown("### Recommendation")
            st.markdown(advice)
        else:
            st.info("Enter quantity & buy price in sidebar for portfolio advice.")

else:
    st.info("Add a stock to watchlist or enter a symbol and click Analyze.")
