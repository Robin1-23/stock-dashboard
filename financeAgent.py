import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
import openai
from phi.model.openai import OpenAIChat

import pandas as pd
import yfinance as yf
import re

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ---------------- CSS ----------------
st.markdown("""
    <style>
    .main { padding: 20px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stTextInput>div>input { border-radius: 5px; }
    h1, h2, h3 { color: #2c3e50; }
    .news-item { margin-bottom: 15px; padding: 10px; border-left: 4px solid #3498db; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
    </style>
""", unsafe_allow_html=True)

# ---------------- Session state for watchlist ----------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ---------------- Sidebar: Watchlist ----------------
with st.sidebar:
    st.title("Watchlist")

    new_symbol = st.text_input(
        "Add symbol",
        value="",
        placeholder="e.g. TCS, INFY, AAPL",
        key="watchlist_input"
    )

    if st.button("Add to watchlist"):
        if new_symbol.strip():
            sym = new_symbol.strip().upper()
            if sym not in st.session_state.watchlist:
                st.session_state.watchlist.append(sym)
            else:
                st.info(f"{sym} is already in your watchlist.")

    st.markdown("### Saved Symbols")
    to_remove = []

    if st.session_state.watchlist:
        for sym in st.session_state.watchlist:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(sym, key=f"wl_{sym}"):
                    st.session_state.selected_from_watchlist = sym
            with col2:
                if st.button("❌", key=f"rm_{sym}"):
                    to_remove.append(sym)
    else:
        st.caption("No symbols yet. Add one above.")

    for sym in to_remove:
        if sym in st.session_state.watchlist:
            st.session_state.watchlist.remove(sym)

# ---------------- App title ----------------
st.title("Stock Analysis Dashboard")
st.markdown("""
Enter an Indian stock ticker (e.g., **TCS**, **RELIANCE**, **INFY**).  
If no exchange suffix is provided, the app will **automatically assume NSE (.NS)**.
""")

# ---------------- Main Inputs ----------------
ticker = st.text_input(
    "Stock Ticker",
    value="TCS",
    placeholder="Enter ticker like TCS, RELIANCE, TCS.NS, TCS.BO, AAPL",
)

submit_button = st.button("Get Stock Info")

# Watchlist click detection
selected_from_watchlist = (
    st.session_state.pop("selected_from_watchlist", None)
    if "selected_from_watchlist" in st.session_state
    else None
)

# ---------------- Helper: normalize ticker ----------------
def normalize_ticker(t: str) -> str:
    if not t:
        return t
    t = t.strip().upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return t
    if "." in t:  # user already provided exchange
        return t
    return t + ".NS"  # default India NSE

# ---------------- Agents ----------------
websearch_agent = Agent(
    name="Web Search",
    role="Search the web",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
    show_tool_calls=False,
    markdown=True,
)

financeAgent = Agent(
    name="Finance Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use markdown tables for data."],
    show_tool_calls=True,
    markdown=True,
)

multiAiAgent = Agent(
    team=[websearch_agent, financeAgent],
    instructions=[
        "Return output in sections:",
        "## Analyst Recommendations",
        "## Latest News",
        "## Sources",
    ],
    show_tool_calls=False,
    markdown=True,
)

# ---------------- Helper: parse markdown table ----------------
def parse_markdown_table(table_text: str):
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]

    header_line = None
    for line in lines:
        if line.startswith("|") and line.count("|") >= 2 and not set(line.replace("|","")).issubset({"-"," "}):
            header_line = line
            break

    if not header_line:
        return None

    headers = [h.strip() for h in header_line.split("|")[1:-1]]

    data_rows = []
    header_found = False
    for line in lines:
        if line == header_line:
            header_found = True
            continue

        if not header_found:
            continue

        clean = line.replace("|", "").replace("-", "").strip()
        if not clean:
            continue

        if line.startswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            data_rows.append(cells)

    if not data_rows:
        return None

    # normalize row lengths
    normalized = []
    for r in data_rows:
        if len(r) < len(headers):
            r += [""] * (len(headers) - len(r))
        normalized.append(r[:len(headers)])

    try:
        return pd.DataFrame(normalized, columns=headers)
    except:
        return None

# ---------------- Helper: key metrics ----------------
def show_key_metrics(info: dict):
    st.subheader("Key Metrics")

    cp = info.get("currentPrice") or info.get("regularMarketPrice")
    pct = info.get("regularMarketChangePercent")
    mc = info.get("marketCap")
    pe = info.get("trailingPE")

    cols = st.columns(4)

    cols[0].metric("Current Price", cp if cp else "N/A")
    cols[1].metric("Day Change", f"{pct:.2f}%" if pct else "N/A")
    cols[2].metric("Market Cap", f"{mc:,}" if isinstance(mc, int) else mc or "N/A")
    cols[3].metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")

# ---------------- Helper: price chart ----------------
def show_price_chart(symbol: str):
    st.subheader("Price History")
    period = st.selectbox(
        "Select Period",
        ["1mo", "3mo", "6mo", "1y", "5y"],
        index=2,
        key=f"price_period_{symbol}",
    )
    data = yf.download(symbol, period=period)
    if not data.empty:
        st.line_chart(data["Close"])
    else:
        st.warning("No price data found.")

# ---------------- Helper: fundamentals ----------------
def show_fundamentals(symbol: str, info: dict):
    st.subheader("Fundamentals")

    fundamentals = {
        "Market Cap": info.get("marketCap"),
        "Enterprise Value": info.get("enterpriseValue"),
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "PEG Ratio": info.get("pegRatio"),
        "Dividend Yield": info.get("dividendYield"),
        "Return on Equity": info.get("returnOnEquity"),
        "Return on Assets": info.get("returnOnAssets"),
        "Book Value": info.get("bookValue"),
        "EPS (TTM)": info.get("trailingEps"),
    }

    df = pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in fundamentals.items() if v is not None]
    )
    st.table(df)

    st.subheader("Annual Financials")
    try:
        fin = yf.Ticker(symbol).financials
        if fin is not None and not fin.empty:
            st.dataframe(fin.T)
        else:
            st.info("Financial data unavailable.")
    except:
        st.error("Failed to load financials.")

# ---------------- Main display function ----------------
def display_stock_info(symbol: str):
    symbol = normalize_ticker(symbol)

    # Tabs (NO COMPARISON TAB)
    tab_overview, tab_fundamentals, tab_analysts, tab_news = st.tabs(
        ["Overview", "Fundamentals", "Analysts", "News"]
    )

    with st.spinner(f"Loading data for {symbol}..."):
        try:
            t = yf.Ticker(symbol)
            try:
                info = t.info
            except:
                info = {}

            # ---- AI response for analysts & news ----
            response = multiAiAgent.run(
                f"Provide analyst recommendations and latest news for {symbol}"
            )
            content = getattr(response, "content", "") or ""

            analyst_match = re.search(
                r"##+\s*Analyst Recommendations\s*(.*?)(?=\n##+|$)",
                content,
                flags=re.DOTALL,
            )
            news_match = re.search(
                r"##+\s*Latest News\s*(.*?)(?=\n##+|$)",
                content,
                flags=re.DOTALL,
            )

            # ---- Overview ----
            with tab_overview:
                st.caption(f"Symbol used: **{symbol}**")
                if info:
                    show_key_metrics(info)
                show_price_chart(symbol)

            # ---- Fundamentals ----
            with tab_fundamentals:
                if info:
                    show_fundamentals(symbol, info)
                else:
                    st.warning("No fundamentals available.")

            # ---- Analysts ----
            with tab_analysts:
                if content:
                    with st.expander("Show raw AI output"):
                        st.markdown(content)

                st.subheader("Analyst Recommendations")
                if analyst_match:
                    table_text = analyst_match.group(1)
                    df = parse_markdown_table(table_text)
                    if df is not None:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.markdown(table_text)
                else:
                    st.info("No analyst data found.")

            # ---- News ----
            with tab_news:
                st.subheader("Latest News")
                if news_match:
                    blocks = news_match.group(1).strip().split("\n- ")
                    for item in blocks:
                        item = item.strip()
                        if item:
                            st.markdown(f"<div class='news-item'>{item}</div>", unsafe_allow_html=True)
                else:
                    st.info("No news data found.")

        except Exception as e:
            st.error(f"Error fetching data: {e}")

# ---------------- Run logic ----------------
if submit_button:
    display_stock_info(ticker)
elif selected_from_watchlist:
    display_stock_info(selected_from_watchlist)
