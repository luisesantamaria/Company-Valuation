# app.py — Company Valuation (robusto para Streamlit Cloud / Yahoo 429)

import os
import time
import random
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import yfinance as yf

from streamlit_searchbox import st_searchbox

# ----------------------- Configuración Streamlit -----------------------
st.set_page_config(page_title="Company Valuation", layout="wide")
alt.data_transformers.disable_max_rows()

# ----------------------- Parámetros globales --------------------------
PRICE_PERIOD = "5y"
PRICE_INTERVAL = "1d"

# Menos peers para reducir presión a la API (antes 20)
MAX_PEERS = 8
MIN_COVERAGE = 0.50

INDEX_BENCH = {
    "SP500": "^GSPC", "S&P500": "^GSPC", "S&P 500": "^GSPC",
    "IPC": "^MXX", "S&P/BMV IPC": "^MXX",
    "IBOV": "^BVSP", "IBOVESPA": "^BVSP", "IBOVESPA INDEX": "^BVSP"
}

# ----------------------- Sesión HTTP robusta --------------------------
# Evita 429 y errores transitorios en entornos compartidos (Streamlit Cloud)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@st.cache_resource(show_spinner=False)
def http_session():
    s = requests.Session()
    retry = Retry(
        total=3,                # hasta 3 reintentos
        backoff_factor=1.2,     # 0s, 1.2s, 2.4s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"
    })
    return s

SESSION = http_session()

def _sleep_jitter(base=0.6):
    """Pequeña pausa con jitter para ser amable con la API."""
    time.sleep(base + random.random()*0.4)  # 0.6–1.0s por defecto

# ----------------------- Utils de formato -----------------------------
def _num(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def _fmt_money(x):
    try: return f"{x:,.0f}"
    except: return "—"

def _fmt_money2(x):
    try: return f"{x:,.2f}"
    except: return "—"

def _fmt_pct(x, d=1):
    try: return f"{x*100:.{d}f}%"
    except: return "—"

def _fmt_date(x):
    try:
        if pd.isna(x): return "—"
        if isinstance(x, (int, float)) and x > 10_000:
            return pd.to_datetime(x, unit="s").date().isoformat()
        if isinstance(x, (pd.Timestamp, datetime, date)):
            return pd.to_datetime(x).date().isoformat()
        return str(x)
    except: return "—"

def _truncate(text: str, max_chars=900):
    if not text: return ""
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "…"

def _prune_statement(df: pd.DataFrame, min_cov=MIN_COVERAGE):
    if df is None or df.empty: return pd.DataFrame()
    keep = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        coverage = s.notna().mean()
        all_nan = s.isna().all()
        all_zero = np.nan_to_num(s.values, nan=0.0).sum() == 0.0
        if (not all_nan) and (not all_zero) and (coverage >= min_cov):
            keep.append(col)
    return df[keep] if keep else pd.DataFrame()

def _ensure_naive_datetime_index(series: pd.Series) -> pd.Series:
    if series is None or series.empty: return series
    idx = pd.to_datetime(series.index)
    try:
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            idx = idx.tz_localize(None)
    except Exception:
        idx = pd.to_datetime([pd.Timestamp(x).to_pydatetime().replace(tzinfo=None) for x in series.index])
    out = series.copy(); out.index = idx
    return out

def _percentile(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or not np.isfinite(value): return np.nan
    return (s <= value).mean()

def _last(series: pd.Series):
    try:
        s = series.dropna()
        return s.iloc[-1] if not s.empty else np.nan
    except: return np.nan

def _avg_last2(series: pd.Series):
    try:
        s = series.dropna()
        if len(s) >= 2: return float(s.iloc[-2:].mean())
        if len(s) == 1: return float(s.iloc[-1])
        return np.nan
    except: return np.nan

# ----------------------- Render helpers -------------------------------
def render_chips(pairs):
    # pairs: [(label, value, formatter_fn), ...]
    valid = [(lab, val, fmt) for lab, val, fmt in pairs if (isinstance(val, (int, float)) and np.isfinite(val))]
    if not valid: return
    cols = st.columns(len(valid))
    for col, (lab, val, fmt) in zip(cols, valid):
        col.metric(lab, fmt(val) if fmt else str(val))

def render_reco_pill(label: str):
    if not label: return
    color = {"BUY": "#12B886", "SELL": "#E03131", "HOLD": "#FAB005"}.get(label, "#4C6EF5")
    st.markdown(
        f"""
        <div style="width:100%; display:flex; justify-content:flex-end; margin: .25rem 0;">
          <span style="
            background: transparent;
            border: 1px solid {color};
            color:{color};
            padding: .35rem .6rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: .9rem;">
            Recommendation: {label}
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def chart_price(series: pd.Series, title="Price"):
    if series is None or series.empty: return None
    df = series.rename("Price").reset_index().rename(columns={"index": "Date"})
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Price:Q", title="Price", scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Price:Q", format=",.2f")]
        )
        .properties(title=title)
        .interactive()
    )

def chart_multi_returns(df: pd.DataFrame, keep_cols, title="Cumulative Return (%)"):
    if df is None or df.empty: return None
    sub = df[keep_cols].dropna(how="all")
    if sub.empty: return None
    sub_ret = sub.copy()
    for c in sub_ret.columns:
        s = sub_ret[c].dropna()
        if s.empty: continue
        sub_ret[c] = s / s.iloc[0] - 1.0
    tidy = sub_ret.reset_index().melt("Date", var_name="Series", value_name="Return")
    return (
        alt.Chart(tidy)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Return:Q", title="Cumulative Return", axis=alt.Axis(format="%"),
                    scale=alt.Scale(zero=False, nice=True)),
            color="Series:N",
            tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Return:Q", format=".1%")]
        )
        .properties(title=title)
        .interactive()
    )

# ----------------------- Monte Carlo (EWMA) ---------------------------
def build_mc_ewma_paths(px_series: pd.Series,
                        business_days: int = 252,
                        n_paths: int = 1500,
                        lam: float = 0.94,
                        quantiles=(0.10, 0.50, 0.90),
                        seed: int = 7) -> Optional[pd.DataFrame]:
    s = px_series.dropna()
    if len(s) < 120:
        return None

    logret = np.log(s).diff().dropna()
    mu_d = float(logret.mean())

    var_ewma = logret.pow(2).ewm(alpha=(1.0 - lam)).mean()
    sigma_t = np.sqrt(var_ewma)
    sigma_last = float(sigma_t.iloc[-1])

    u = (logret / sigma_t).replace([np.inf, -np.inf], np.nan).dropna().values
    if len(u) < 50 or not np.isfinite(sigma_last) or sigma_last <= 0:
        return None

    rng = np.random.default_rng(seed)
    Z = rng.choice(u, size=(business_days, n_paths), replace=True)
    R = mu_d + sigma_last * Z

    S0 = float(s.iloc[-1])
    log_paths = np.log(S0) + np.cumsum(R, axis=0)
    P = np.exp(log_paths)

    dates = pd.bdate_range(s.index[-1] + pd.Timedelta(days=1), periods=business_days)

    terminal = P[-1, :]
    qs = np.quantile(terminal, quantiles)

    chosen_idx = []
    for q in qs:
        order = np.argsort(np.abs(terminal - q))
        idx_sel = next((i for i in order if i not in chosen_idx), order[0])
        chosen_idx.append(idx_sel)

    labels = ["Bear", "Base", "Bull"]
    rows = []
    for lab, j in zip(labels, chosen_idx):
        rows.extend({"Date": dt, "Scenario": lab, "Price": float(p)} for dt, p in zip(dates, P[:, j]))

    return pd.DataFrame(rows)

def chart_hist_plus_scenarios(hist_series: pd.Series, scen_df: pd.DataFrame,
                              title="Price: 5Y History + 1Y Scenarios"):
    hist_df = hist_series.dropna().rename("Price").reset_index()
    hist_df["Scenario"] = "Historical"
    hist_df = hist_df.rename(columns={hist_df.columns[0]: "Date"})
    tidy = pd.concat([hist_df[["Date","Scenario","Price"]], scen_df[["Date","Scenario","Price"]]],
                     axis=0, ignore_index=True)
    color = alt.condition(
        alt.datum.Scenario == "Historical",
        alt.value("#666"),
        alt.Color("Scenario:N")
    )
    return (
        alt.Chart(tidy)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Price:Q", title="Price", scale=alt.Scale(zero=False, nice=True)),
            color=color,
            tooltip=[alt.Tooltip("Date:T"), "Scenario:N", alt.Tooltip("Price:Q", format=",.2f")]
        )
        .properties(title=title)
        .interactive()
    )

# ----------------------- Carga de compañías ---------------------------
@st.cache_data(show_spinner=False)
def load_companies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    for req in ["ticker", "name", "index", "sector"]:
        if req not in cols_lower:
            raise ValueError(f"Column '{req}' not found in companies_list.csv")
    df = df.rename(columns={
        cols_lower["ticker"]: "ticker",
        cols_lower["name"]: "name",
        cols_lower["index"]: "index",
        cols_lower["sector"]: "sector",
    })
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["index"] = df["index"].astype(str).str.upper().str.strip()
    df["sector"] = df["sector"].astype(str).str.strip()
    return df

# ----------------------- Wrappers robustos de Yahoo -------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_history(symbol: str, period=PRICE_PERIOD, interval=PRICE_INTERVAL) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = yf.download(
                symbol, period=period, interval=interval,
                auto_adjust=True, progress=False, session=SESSION, threads=False
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                out = df.loc[:, ["Close"]].copy()
                out.columns = [symbol]
                return out
        except Exception:
            pass
        _sleep_jitter(0.8 * (attempt + 1))
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def get_multi_history(symbols, period=PRICE_PERIOD, interval=PRICE_INTERVAL) -> pd.DataFrame:
    syms = [s for s in symbols if isinstance(s, str) and s]
    if not syms: return pd.DataFrame()
    # Intento 1: descarga en bloque
    for attempt in range(2):
        try:
            df = yf.download(
                syms, period=period, interval=interval,
                auto_adjust=True, progress=False, session=SESSION, threads=False
            )
            close = df["Close"] if "Close" in df else df
            if isinstance(close, pd.Series):
                close = close.to_frame()
            if isinstance(close.columns, pd.MultiIndex):
                close.columns = [c[-1] for c in close.columns]
            else:
                close.columns = [str(c) for c in close.columns]
            out = close.dropna(how="all")
            if not out.empty:
                return out
        except Exception:
            pass
        _sleep_jitter(0.9 * (attempt + 1))
    # Fallback: de a uno
    frames = []
    for s in syms[:MAX_PEERS]:
        h = get_history(s, period, interval)
        if not h.empty: frames.append(h)
        _sleep_jitter(0.35)
    return pd.concat(frames, axis=1).dropna(how="all") if frames else pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=7200)
def get_info(tick: str) -> dict:
    for attempt in range(3):
        try:
            # Preferir .get_info() (más estable en yfinance recientes)
            return yf.Ticker(tick, session=SESSION).get_info() or {}
        except Exception:
            _sleep_jitter(0.9 * (attempt + 1))
    return {}

@st.cache_data(show_spinner=False, ttl=7200)
def get_statements(tick: str, last_years: int = 4):
    def _prep(df):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty): return pd.DataFrame()
        out = df.T.copy()
        try: out.index = pd.to_datetime(out.index).year
        except: pass
        out = out.sort_index().tail(last_years)
        for c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    # Intentos con sesión compartida
    for attempt in range(3):
        try:
            t = yf.Ticker(tick, session=SESSION)
            fin = _prep(t.financials)
            bs  = _prep(t.balance_sheet)
            cf  = _prep(t.cashflow)
            break
        except Exception:
            fin = bs = cf = pd.DataFrame()
            _sleep_jitter(1.0 * (attempt + 1))

    def pick(df, names):
        if df is None or df.empty: return pd.Series(dtype=float)
        for n in names:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce")
        return pd.Series([np.nan]*len(df), index=df.index, dtype=float)

    revenue   = pick(fin, ["Total Revenue","Revenue"])
    cogs      = pick(fin, ["Cost Of Revenue","Cost of Revenue"])
    gross     = pick(fin, ["Gross Profit"])
    ebit      = pick(fin, ["Operating Income","OperatingIncome","EBIT"])
    ebitda    = pick(fin, ["EBITDA"])
    da        = pick(fin, ["Reconciled Depreciation","Depreciation & Amortization","Depreciation"])
    netinc    = pick(fin, ["Net Income","NetIncome"])
    if gross.notna().sum()==0 and (revenue.notna().sum()>0 and cogs.notna().sum()>0):
        gross = revenue - cogs
    if ebit.notna().sum()==0 and (ebitda.notna().sum()>0 and da.notna().sum()>0):
        ebit = ebitda - da
    if ebitda.notna().sum()==0 and (ebit.notna().sum()>0 and da.notna().sum()>0):
        ebitda = ebit + da

    income_tbl = pd.DataFrame({
        "Revenue": revenue, "Gross Profit": gross,
        "Operating Income": ebit, "EBITDA": ebitda, "Net Income": netinc
    }).round(0)

    tot_assets = pick(bs, ["Total Assets"])
    tot_liab   = pick(bs, ["Total Liabilities Net Minority Interest","Total Liabilities"])
    equity     = pick(bs, ["Total Stockholder Equity","Stockholders Equity"])
    cash_sti   = pick(bs, ["Cash Cash Equivalents And Short Term Investments","Cash And Cash Equivalents"])
    short_debt = pick(bs, ["Short Long Term Debt"])
    long_debt  = pick(bs, ["Long Term Debt","Long Term Debt And Capital Lease Obligation"])
    tot_debt   = pick(bs, ["Total Debt"])
    if tot_debt.notna().sum()==0:
        tot_debt = short_debt.add(long_debt, fill_value=0.0)
    cur_assets = pick(bs, ["Total Current Assets","Current Assets"])
    cur_liab   = pick(bs, ["Total Current Liabilities","Current Liabilities"])
    net_ppe    = pick(bs, ["Net PPE","Property Plant Equipment Net","Property Plant & Equipment Net"])

    balance_tbl = pd.DataFrame({
        "Total Assets": tot_assets, "Total Liabilities": tot_liab, "Equity": equity,
        "Cash & ST Inv.": cash_sti, "Total Debt": tot_debt,
        "Current Assets": cur_assets, "Current Liabilities": cur_liab, "Net PPE": net_ppe
    }).round(0)

    cfo   = pick(cf, ["Operating Cash Flow","Total Cash From Operating Activities"])
    capex = -pick(cf, ["Capital Expenditure","Investments"]).fillna(0)
    fcf   = pick(cf, ["Free Cash Flow"])
    if fcf.notna().sum()==0 and (cfo.notna().sum()>0):
        fcf = (cfo - capex).rename("Free Cash Flow (derived)")
    cashflow_tbl = pd.DataFrame({"CFO": cfo, "CAPEX": capex, "FCF": fcf}).round(0)

    income_tbl   = _prune_statement(income_tbl, MIN_COVERAGE).dropna(how="all")
    balance_tbl  = _prune_statement(balance_tbl, MIN_COVERAGE).dropna(how="all")
    cashflow_tbl = _prune_statement(cashflow_tbl, MIN_COVERAGE).dropna(how="all")
    return income_tbl, balance_tbl, cashflow_tbl

# ----------------------- Fair price y recomendación -------------------
def compute_fair_price_band(info: dict, cmp_df: pd.DataFrame):
    ev = _num(info.get("enterpriseValue"))
    ebitda = _num(info.get("ebitda"))
    revenue = _num(info.get("totalRevenue"))
    eps_ttm = _num(info.get("trailingEps"))
    total_debt = _num(info.get("totalDebt"), 0.0)
    cash = _num(info.get("cash", info.get("totalCash")), 0.0)
    shares = _num(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))

    bands = {"low": [], "mid": [], "high": []}
    disp_list, peers_n_list, used = [], [], []

    def add_method(colname, base_val, is_ev=True):
        nonlocal bands, disp_list, peers_n_list, used
        if colname not in cmp_df.columns: return
        s = pd.to_numeric(cmp_df[colname], errors="coerce").dropna()
        if s.empty or not np.isfinite(base_val) or not np.isfinite(shares) or shares <= 0: return
        p25, p50, p75 = np.percentile(s.values, [25, 50, 75])
        if np.isfinite(p50) and p50 != 0:
            disp_list.append((p75 - p25) / abs(p50))
        peers_n_list.append(len(s))
        if is_ev:
            low_eq = p25 * base_val - total_debt + cash
            mid_eq = p50 * base_val - total_debt + cash
            high_eq = p75 * base_val - total_debt + cash
        else:
            low_eq = p25 * base_val
            mid_eq = p50 * base_val
            high_eq = p75 * base_val
        bands["low"].append(low_eq / shares)
        bands["mid"].append(mid_eq / shares)
        bands["high"].append(high_eq / shares)
        used.append(colname)

    if np.isfinite(ebitda) and ebitda > 0: add_method("EV/EBITDA", ebitda, True)
    if np.isfinite(revenue) and revenue > 0: add_method("EV/Sales", revenue, True)
    if np.isfinite(eps_ttm) and eps_ttm > 0: add_method("P/E", eps_ttm, False)

    if not bands["mid"]: return None
    low = float(np.nanmedian(bands["low"]))
    mid = float(np.nanmedian(bands["mid"]))
    high = float(np.nanmedian(bands["high"]))
    peers_n = int(np.nanmedian(peers_n_list)) if peers_n_list else 0
    disp = float(np.nanmedian(disp_list)) if disp_list else np.nan
    confidence = "HIGH" if (peers_n >= 8 and np.isfinite(disp) and disp < 0.35) else ("MEDIUM" if (peers_n >= 5 and np.isfinite(disp) and disp < 0.60) else "LOW")
    return {"low": low, "mid": mid, "high": high, "used": used, "peers_n": peers_n, "disp": disp, "confidence": confidence}

def compute_reco(price_now, fair_dict):
    if not (np.isfinite(price_now) and fair_dict and np.isfinite(fair_dict.get("mid", np.nan))):
        return None
    upside = fair_dict["mid"] / price_now - 1.0
    return "BUY" if upside >= 0.15 else ("SELL" if upside <= -0.15 else "HOLD")

# ----------------------- Carga universo -------------------------------
companies_path = os.path.join("data", "companies_list.csv")
if not os.path.exists(companies_path):
    st.error("Missing data file: data/companies_list.csv"); st.stop()

companies = load_companies(companies_path)
TICKER_TO_NAME = dict(zip(companies["ticker"], companies["name"]))

st.subheader("Company Valuation")

def search_fn(query: str):
    q = (query or "").strip()
    if not q:
        df = companies.head(10)
    else:
        df = companies[
            companies["name"].str.contains(q, case=False, na=False) |
            companies["ticker"].str.contains(q.upper(), na=False)
        ].head(10)
    return [(f"{r.name} — {r.ticker}", r.ticker) for r in df.itertuples(index=False)]

chosen_ticker = st_searchbox(search_fn, key="searchbox", placeholder="Type a company name or ticker")
ticker = chosen_ticker

if not ticker:
    st.info("Start typing above and pick a suggestion to analyze.")
    st.stop()

sel = companies.loc[companies["ticker"] == ticker]
if sel.empty:
    st.error("Selected ticker not found in the universe file."); st.stop()

comp_name = sel["name"].values[0]
comp_index = sel["index"].values[0]
comp_sector = sel["sector"].values[0]
bench = INDEX_BENCH.get(comp_index.upper(), None)

# ----------------------- Info y peers -------------------------------
with st.spinner("Loading company info…"):
    info = get_info(ticker)
    peers = companies[(companies["index"].str.upper() == comp_index.upper()) &
                      (companies["sector"] == comp_sector)]
    peers = peers[peers["ticker"] != ticker].head(MAX_PEERS)
    peer_ticks = peers["ticker"].tolist()

    @st.cache_data(show_spinner=False, ttl=1800)
    def fetch_peer_row(tk):
        i = get_info(tk)  # cacheado
        ev = _num(i.get("enterpriseValue"))
        ebitda = _num(i.get("ebitda"))
        rev = _num(i.get("totalRevenue"))
        pe = _num(i.get("trailingPE"))
        ev_ebitda = ev/ebitda if (np.isfinite(ev) and np.isfinite(ebitda) and ebitda>0) else np.nan
        ev_sales  = ev/rev    if (np.isfinite(ev) and np.isfinite(rev)    and rev>0)    else np.nan
        return {"ticker": tk, "EV/EBITDA": ev_ebitda, "EV/Sales": ev_sales, "P/E": pe}

    rows = []
    for tk in [ticker] + peer_ticks[:MAX_PEERS]:
        rows.append(fetch_peer_row(tk))
        _sleep_jitter(0.25)   # reduce presión
    cmp_df = pd.DataFrame(rows)
    cmp_df["Company"] = cmp_df["ticker"].map(TICKER_TO_NAME).fillna(cmp_df["ticker"])
    cmp_df = cmp_df.drop(columns=["ticker"]).set_index("Company")

# ----------------------- Header métricas -----------------------------
col1, col2, col3, col4 = st.columns([2, 2.8, 1.2, 1.0])
with col1:
    st.subheader(f"{comp_name} ({ticker})")
    st.caption(f"Index: {comp_index}  •  Sector: {comp_sector}")
with col2:
    mc = _num(info.get("marketCap"))
    st.metric("Market Cap", _fmt_money(mc))
with col3:
    price_now = np.nan
    hist_now = get_history(ticker, period="1mo", interval="1d")
    if not hist_now.empty:
        price_now = float(hist_now.iloc[-1, 0])
    st.metric("Current Price", _fmt_money2(price_now) if np.isfinite(price_now) else "—")
with col4:
    st.metric("Currency", info.get("currency", "—"))

fair_global = None
try:
    fair_global = compute_fair_price_band(info, cmp_df)
except Exception:
    fair_global = None
reco_label = compute_reco(price_now, fair_global)

# ----------------------- Tabs -------------------------------
tabs = st.tabs([
    "Overview", "Price & Benchmark", "Peers & Sector",
    "Financials", "Fair Price", "Dividends",
    "Advanced", "Notes & Data"
])

with tabs[0]:
    with st.spinner("Fetching 1Y price…"):
        hist_1y = get_history(ticker, period="1y", interval="1d")
    if not hist_1y.empty:
        s = hist_1y.iloc[:, 0].dropna(); n = len(s)
        r1m = s.iloc[-1] / s.iloc[-22] - 1.0 if n > 21 else np.nan
        r3m = s.iloc[-1] / s.iloc[-64] - 1.0 if n > 63 else np.nan
        try:
            year_start = pd.Timestamp(datetime.utcnow().year, 1, 1)
            idx_pos = np.searchsorted(s.index.values, np.datetime64(year_start))
            rytd = s.iloc[-1] / float(s.iloc[idx_pos]) - 1.0 if idx_pos < n else np.nan
        except Exception:
            rytd = np.nan
        r1y = s.iloc[-1] / s.iloc[0] - 1.0
        base = (s / s.iloc[0]) * 100.0
        dd1y = float((base / base.cummax() - 1.0).min())
    else:
        r1m = r3m = rytd = r1y = dd1y = np.nan

    render_chips([
        ("1M", r1m, _fmt_pct),
        ("3M", r3m, _fmt_pct),
        ("YTD", rytd, _fmt_pct),
        ("1Y", r1y, _fmt_pct),
        ("Max DD (1Y)", dd1y, _fmt_pct),
    ])

    left, right = st.columns(2, gap="small")
    with left:
        st.subheader("Company Profile")
        summary = info.get("longBusinessSummary") or ""
        parts = [info.get("city", ""), info.get("state", ""), info.get("country", "")]
        hq = ", ".join([p for p in parts if p])
        employees = _num(info.get("fullTimeEmployees"))
        website = info.get("website")
        founded = info.get("founded")
        ipo_epoch = info.get("firstTradeDateEpochUtc")
        ipo_year = pd.to_datetime(ipo_epoch, unit="s").year if isinstance(ipo_epoch, (int, float)) else None
        bullets = []
        if hq: bullets.append(f"- **Headquarters:** {hq}")
        if founded or ipo_year: bullets.append(f"- **Founded / IPO year:** {founded or ipo_year}")
        if np.isfinite(employees): bullets.append(f"- **Employees:** {int(employees):,}")
        if website: bullets.append(f"- **Website:** [{website}]({website})")
        if summary: st.write(_truncate(summary, 900))
        if bullets: st.markdown("\n".join(bullets))
        if not summary and not bullets: st.caption("No profile details available.")
        px_date = _fmt_date(hist_1y.index.max()) if not hist_1y.empty else "—"
        fy_end = _fmt_date(info.get("lastFiscalYearEnd")); mrq = _fmt_date(info.get("mostRecentQuarter"))
        st.caption(f"**Data Freshness** — Prices through {px_date}. Fundamentals: FY end {fy_end} • MRQ {mrq}.")
    with right:
        render_reco_pill(reco_label)
        if not hist_1y.empty:
            ch = chart_price(hist_1y.iloc[:, 0], title=f"{ticker} — 1Y Price")
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("Price data unavailable right now.")

with tabs[1]:
    with st.spinner("Fetching prices & peers…"):
        series = [ticker] + ([bench] if bench else [])
        price_df = get_multi_history(series + peer_ticks, period=PRICE_PERIOD, interval=PRICE_INTERVAL)
    try:
        keep_cols = [ticker] + ([bench] if bench else [])
        px_sub = price_df[keep_cols].dropna(how="any").copy()
        px_1y = px_sub.iloc[-252:] if len(px_sub) > 252 else px_sub
        ret = px_1y.pct_change().dropna()
        if not ret.empty:
            try:
                rf_hist = get_history("^IRX", period="6mo", interval="1d")
                rf_ann = float(rf_hist.iloc[-1, 0]) / 100.0 if not rf_hist.empty else 0.0
            except Exception:
                rf_ann = 0.0
            rf_daily = (1 + rf_ann)**(1/252) - 1
            ann_vol = ret[ticker].std() * np.sqrt(252)
            ann_ret = (1 + ret[ticker]).prod()**(252/len(ret)) - 1
            sharpe = (ret[ticker].mean() - rf_daily) / (ret[ticker].std() + 1e-12) * np.sqrt(252)
            beta = np.nan
            if bench and bench in ret.columns:
                cov = ret[[ticker, bench]].cov().iloc[0, 1]
                varb = ret[bench].var()
                if varb > 0: beta = cov / varb
            render_chips([
                ("Ann. Return (1Y)", ann_ret, _fmt_pct),
                ("Ann. Vol (1Y)", ann_vol, _fmt_pct),
                ("Sharpe (1Y)", sharpe, lambda v: f"{v:.2f}"),
                ("Beta vs Bench (1Y)", beta, lambda v: f"{v:.2f}") if np.isfinite(beta) else None,
            ])
    except Exception:
        pass

    if not price_df.empty:
        df = price_df.copy()
        peer_cols = [c for c in peer_ticks if c in df.columns]
        if peer_cols:
            sector_curve = df[peer_cols].dropna(how="all").pct_change().add(1).cumprod()
            if not sector_curve.empty:
                df["Sector Avg"] = sector_curve.mean(axis=1)
        keep_cols = [c for c in [ticker, "Sector Avg", bench] if c and c in df.columns]
        ch = chart_multi_returns(df, keep_cols, title="Cumulative Return (%)")
        if ch: st.altair_chart(ch, use_container_width=True)
        else: st.info("Not enough overlapping price history to plot.")
    else:
        st.info("Price data unavailable right now.")

with tabs[2]:
    my = cmp_df.loc[comp_name] if comp_name in cmp_df.index else pd.Series(dtype=float)
    ev_ebitda_pct = _percentile(cmp_df["EV/EBITDA"], _num(my.get("EV/EBITDA"))) if "EV/EBITDA" in cmp_df else np.nan
    ev_sales_pct  = _percentile(cmp_df["EV/Sales"],  _num(my.get("EV/Sales")))  if "EV/Sales"  in cmp_df else np.nan
    pe_pct        = _percentile(cmp_df["P/E"],       _num(my.get("P/E")))       if "P/E"       in cmp_df else np.nan
    render_chips([
        ("EV/EBITDA pctile", ev_ebitda_pct, _fmt_pct),
        ("EV/Sales pctile",  ev_sales_pct,  _fmt_pct),
        ("P/E pctile",       pe_pct,        _fmt_pct),
    ])
    st.write("**Comparable Multiples (clean)**")
    median_row = cmp_df.median(numeric_only=True).to_frame("Sector Median").T
    table = pd.concat([cmp_df.round(2), median_row.round(2)])
    st.dataframe(table, use_container_width=True)
    st.caption(f"Peers = companies in {comp_index}, sector {comp_sector}. Row for: {comp_name}.")

with tabs[3]:
    with st.spinner("Fetching financial statements…"):
        inc, bal, cfs = get_statements(ticker, last_years=4)
    revenue = inc["Revenue"] if "Revenue" in inc else pd.Series(dtype=float)
    gross   = inc["Gross Profit"] if "Gross Profit" in inc else pd.Series(dtype=float)
    ebit    = inc["Operating Income"] if "Operating Income" in inc else pd.Series(dtype=float)
    ebitda  = inc["EBITDA"] if "EBITDA" in inc else pd.Series(dtype=float)
    netinc  = inc["Net Income"] if "Net Income" in inc else pd.Series(dtype=float)
    assets  = bal["Total Assets"] if "Total Assets" in bal else pd.Series(dtype=float)
    equity  = bal["Equity"] if "Equity" in bal else pd.Series(dtype=float)
    cash    = bal["Cash & ST Inv."] if "Cash & ST Inv." in bal else pd.Series(dtype=float)
    debt    = bal["Total Debt"] if "Total Debt" in bal else pd.Series(dtype=float)
    cur_assets = bal["Current Assets"] if "Current Assets" in bal else pd.Series(dtype=float)
    cur_liab   = bal["Current Liabilities"] if "Current Liabilities" in bal else pd.Series(dtype=float)
    gm_last   = (_last(gross) / _last(revenue)) if np.isfinite(_last(gross)) and np.isfinite(_last(revenue)) and _last(revenue) != 0 else np.nan
    em_last   = (_last(ebitda) / _last(revenue)) if np.isfinite(_last(ebitda)) and np.isfinite(_last(revenue)) and _last(revenue) != 0 else np.nan
    om_last   = (_last(ebit) / _last(revenue))   if np.isfinite(_last(ebit)) and np.isfinite(_last(revenue)) and _last(revenue) != 0 else np.nan
    nm_last   = (_last(netinc) / _last(revenue)) if np.isfinite(_last(netinc)) and np.isfinite(_last(revenue)) and _last(revenue) != 0 else np.nan
    roe_last  = (_last(netinc) / _avg_last2(equity)) if np.isfinite(_last(netinc)) and np.isfinite(_avg_last2(equity)) and _avg_last2(equity) != 0 else np.nan
    roa_last  = (_last(netinc) / _avg_last2(assets)) if np.isfinite(_last(netinc)) and np.isfinite(_avg_last2(assets)) and _avg_last2(assets) != 0 else np.nan
    tax_rate_est = 0.25
    inv_cap_series = debt.add(equity, fill_value=0).sub(cash, fill_value=0)
    roic_last = ((_last(ebit) * (1 - tax_rate_est)) / _avg_last2(inv_cap_series)) if np.isfinite(_last(ebit)) and np.isfinite(_avg_last2(inv_cap_series)) and _avg_last2(inv_cap_series) > 0 else np.nan
    curr_ratio = (_last(cur_assets) / _last(cur_liab)) if np.isfinite(_last(cur_assets)) and np.isfinite(_last(cur_liab)) and _last(cur_liab) != 0 else np.nan
    quick_ratio = ((_last(cur_assets) - 0.0) / _last(cur_liab)) if np.isfinite(_last(cur_assets)) and np.isfinite(_last(cur_liab)) and _last(cur_liab) != 0 else np.nan
    nd_ebitda = ((_last(debt) - _last(cash)) / _last(ebitda)) if np.isfinite(_last(debt)) and np.isfinite(_last(cash)) and np.isfinite(_last(ebitda)) and _last(ebitda) > 0 else np.nan
    render_chips([
        ("Gross Margin", gm_last, _fmt_pct),
        ("EBITDA Margin", em_last, _fmt_pct),
        ("Operating Margin", om_last, _fmt_pct),
        ("Net Margin", nm_last, _fmt_pct),
        ("ROE", roe_last, _fmt_pct),
        ("ROA", roa_last, _fmt_pct),
        ("ROIC (est.)", roic_last, _fmt_pct),
        ("Current Ratio", curr_ratio, _fmt_money2),
        ("Quick Ratio", quick_ratio, _fmt_money2),
        ("Net Debt / EBITDA", nd_ebitda, lambda v: f"{v:.1f}x"),
    ])
    st.caption("Ratios use latest FY; denominators = avg last 2 FYs.")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Income Statement (last 4 FY)**")
        if not inc.empty:
            inc_show = inc.T.copy()
            inc_show = inc_show.reindex(sorted(inc_show.columns), axis=1)
            st.dataframe(inc_show.applymap(_fmt_money), use_container_width=True)
        else:
            st.info("Income Statement not available.")
        st.write("**Cash Flow (last 4 FY)**")
        if not cfs.empty:
            cfs_show = cfs.T.copy()
            cfs_show = cfs_show.reindex(sorted(cfs_show.columns), axis=1)
            st.dataframe(cfs_show.applymap(_fmt_money), use_container_width=True)
        else:
            st.info("Cash Flow not available.")
    with c2:
        st.write("**Balance Sheet (last 4 FY)**")
        if not bal.empty:
            bal_show = bal.T.copy()
            bal_show = bal_show.reindex(sorted(bal_show.columns), axis=1)
            st.dataframe(bal_show.applymap(_fmt_money), use_container_width=True)
        else:
            st.info("Balance Sheet not available.")

with tabs[4]:
    st.write("**Fair Price (comparables-based) + 5Y History & 1Y Scenarios**")

    top_left, top_right = st.columns([3, 1], gap="small")
    with top_left:
        if fair_global:
            render_chips([
                ("Fair (Mid)", fair_global["mid"], _fmt_money2),
                ("Low (P25)", fair_global["low"], _fmt_money2),
                ("High (P75)", fair_global["high"], _fmt_money2),
                ("Upside vs Current", fair_global["mid"]/price_now - 1.0 if np.isfinite(price_now) and price_now>0 else np.nan, _fmt_pct),
            ])
            used = fair_global.get("used") or []
            st.caption("Methods used (sector distributions): " + ", ".join(used) if used else "—")
        else:
            st.info("Not enough peer data to compute a Fair Price band.")
    with top_right:
        render_reco_pill(reco_label)

    bottom_left, bottom_right = st.columns([3, 1], gap="small")
    with bottom_left:
        with st.spinner("Building 1Y scenarios from 5Y history…"):
            px5 = get_history(ticker, period="5y", interval="1d")
        if not px5.empty:
            scn_df = build_mc_ewma_paths(
                px5.iloc[:, 0],
                business_days=252,
                n_paths=1500,
                lam=0.94,
                quantiles=(0.10, 0.50, 0.90),
                seed=7
            )
            if scn_df is not None:
                ch = chart_hist_plus_scenarios(px5.iloc[:, 0], scn_df,
                                               title="5Y History (solid) + 1Y Scenarios (Bear/Base/Bull)")
                st.altair_chart(ch, use_container_width=True)
                bull_1y = float(scn_df.loc[scn_df["Scenario"]=="Bull","Price"].iloc[-1])
                base_1y = float(scn_df.loc[scn_df["Scenario"]=="Base","Price"].iloc[-1])
                bear_1y = float(scn_df.loc[scn_df["Scenario"]=="Bear","Price"].iloc[-1])
            else:
                st.info("Not enough history to build scenarios.")
                bull_1y = base_1y = bear_1y = np.nan
        else:
            st.info("Price history unavailable for scenarios.")
            bull_1y = base_1y = bear_1y = np.nan

    with bottom_right:
        rows = [
            ("Current Price", price_now),
            ("Fair Low (P25)", fair_global["low"] if fair_global else np.nan),
            ("Fair Mid (P50)", fair_global["mid"] if fair_global else np.nan),
            ("Fair High (P75)", fair_global["high"] if fair_global else np.nan),
            ("Bear 1Y (P10)", bear_1y),
            ("Base 1Y (P50)", base_1y),
            ("Bull 1Y (P90)", bull_1y),
        ]
        df_sum = pd.DataFrame(rows, columns=["Metric","Value"])
        df_sum["Value"] = df_sum["Value"].apply(_fmt_money2)
        st.dataframe(df_sum, hide_index=True, use_container_width=True)
        if fair_global and np.isfinite(price_now) and price_now>0:
            up = fair_global["mid"]/price_now - 1.0
            st.caption(f"Implied upside to Fair (Mid): **{_fmt_pct(up)}**")

with tabs[5]:
    try:
        with st.spinner("Fetching dividends…"):
            t = yf.Ticker(ticker, session=SESSION); div = t.dividends
        if div is not None and not div.empty:
            div = _ensure_naive_datetime_index(div)

            now = pd.Timestamp.utcnow().replace(tzinfo=None)
            last_year = now - pd.Timedelta(days=365)
            div_ttm_ps = div[div.index >= last_year].sum() if not div.empty else np.nan  # DPS TTM

            price_now_ok = price_now if np.isfinite(price_now) else np.nan
            yld = (div_ttm_ps / price_now_ok) if (np.isfinite(div_ttm_ps) and np.isfinite(price_now_ok) and price_now_ok > 0) else np.nan

            peer_yields = []
            for tk in peer_ticks[:MAX_PEERS]:
                try:
                    d = yf.Ticker(tk, session=SESSION).dividends
                    if d is None or d.empty: continue
                    d = _ensure_naive_datetime_index(d)
                    d_ttm = d[d.index >= last_year].sum()
                    p = get_history(tk, "1mo", "1d"); px = float(p.iloc[-1, 0]) if not p.empty else np.nan
                    y = (d_ttm / px) if (np.isfinite(d_ttm) and np.isfinite(px) and px > 0) else np.nan
                    if np.isfinite(y): peer_yields.append(y)
                except Exception:
                    pass
                _sleep_jitter(0.15)
            y_median = float(np.nanmedian(peer_yields)) if peer_yields else np.nan

            incL, balL, cfsL = get_statements(ticker, last_years=4)
            fcf_latest = float(cfsL["FCF"].dropna().iloc[-1]) if (not cfsL.empty and "FCF" in cfsL and not cfsL["FCF"].dropna().empty) else np.nan
            ni_latest  = float(incL["Net Income"].dropna().iloc[-1]) if (not incL.empty and "Net Income" in incL and not incL["Net Income"].dropna().empty) else np.nan

            shares   = _num(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
            eps_ttm  = _num(info.get("trailingEps"))

            payout_eps = (div_ttm_ps / eps_ttm) if (np.isfinite(div_ttm_ps) and np.isfinite(eps_ttm) and eps_ttm > 0) else np.nan
            total_div_ttm = (div_ttm_ps * shares) if (np.isfinite(div_ttm_ps) and np.isfinite(shares) and shares > 0) else np.nan
            payout_fcf = (total_div_ttm / fcf_latest) if (np.isfinite(total_div_ttm) and np.isfinite(fcf_latest) and fcf_latest > 0) else np.nan

            render_chips([
                ("Dividend Yield (TTM)", yld, _fmt_pct),
                ("Sector Median (TTM)", y_median, _fmt_pct),
                ("Payout (EPS TTM)",    payout_eps, _fmt_pct),
                ("Payout / FCF (FY)",   payout_fcf, _fmt_pct),
            ])

            monthly = div.resample("M").sum().rename("Dividends").reset_index()
            if not monthly.empty:
                last_date = monthly["Date"].max()
                cutoff = last_date - pd.DateOffset(years=4)
                mon4 = monthly[monthly["Date"] >= cutoff].copy()
                if not mon4.empty:
                    mon4["Year"] = mon4["Date"].dt.year.astype(str)
                    mon4["Month"] = mon4["Date"].dt.strftime("%b")
                    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                    ch_div = (
                        alt.Chart(mon4)
                        .mark_bar()
                        .encode(
                            x=alt.X("Month:N", title=None, sort=month_order),
                            y=alt.Y("Dividends:Q", title="Dividends", scale=alt.Scale(zero=True, nice=True)),
                            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Dividends:Q", format=",.4f")]
                        )
                        .properties(height=200)
                        .facet(column=alt.Column("Year:N", title=None, header=alt.Header(labelOrient="bottom")))
                    )
                    st.altair_chart(ch_div, use_container_width=True)
                else:
                    st.info("No monthly dividends in the last 4 years.")
            else:
                st.info("No dividend history available.")

            if np.isfinite(payout_eps):
                if payout_eps > 1.05:
                    st.warning("Payout (EPS TTM) > 100% — check for specials or one-offs.")
                elif payout_eps > 0.80:
                    st.warning("Payout (EPS TTM) looks stretched.")
        else:
            st.info("Dividend data unavailable.")
    except Exception:
        st.info("Dividend data unavailable.")

with tabs[6]:
    st.subheader("Risk Metrics")
    with st.spinner("Computing risk metrics…"):
        hist3 = get_history(ticker, period="3y", interval="1d")
    if not hist3.empty:
        ret = hist3.iloc[:, 0].pct_change().dropna()
        if not ret.empty:
            vol_annual = ret.std() * np.sqrt(252.0); var95 = np.quantile(ret, 0.05)
            render_chips([
                ("Annualized Volatility", vol_annual, _fmt_pct),
                ("1-day VaR (95%)", -var95, _fmt_pct),
            ])
        else:
            st.info("Not enough return history to compute risk metrics.")
    else:
        st.info("Not enough data to compute risk metrics.")

    st.subheader("Rolling Relationships vs Benchmark")
    if bench:
        with st.spinner("Computing rolling correlation & beta…"):
            px = get_multi_history([ticker, bench], period="3y", interval="1d")
        if not px.empty and all(col in px.columns for col in [ticker, bench]):
            r = px[[ticker, bench]].pct_change().dropna()
            if not r.empty:
                corr90 = r[ticker].rolling(90).corr(r[bench]).dropna().rename("90-day Correlation")
                st.markdown("**90-day Correlation vs Benchmark**")
                st.line_chart(corr90)
                cov = r[ticker].rolling(90).cov(r[bench]); var_b = r[bench].rolling(90).var()
                beta90 = (cov / var_b).dropna().rename("90-day Beta")
                st.markdown("**90-day Beta vs Benchmark**")
                st.line_chart(beta90)
            else: st.info("Not enough overlapping returns for rolling metrics.")
        else:
            st.info("Benchmark or overlapping price data unavailable.")
    else:
        st.info("No benchmark mapping for this index to compute rolling metrics.")

    st.subheader("Drawdown (5Y)")
    with st.spinner("Computing drawdown…"):
        hist5 = get_history(ticker, period="5y", interval="1d")
    if not hist5.empty:
        s = hist5.iloc[:, 0].dropna()
        base = s / s.cummax() - 1.0
        dd_df = base.rename("Drawdown").to_frame().reset_index().rename(columns={"index": "Date"})
        ch_dd = (
            alt.Chart(dd_df)
            .mark_area(opacity=0.6)
            .encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Drawdown:Q", title="Drawdown", axis=alt.Axis(format="%"),
                        scale=alt.Scale(zero=False, nice=True)),
                tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Drawdown:Q", format=".1%")]
            )
            .properties(height=220)
        )
        st.altair_chart(ch_dd, use_container_width=True)
        st.caption(f"Max Drawdown (5Y): **{_fmt_pct(base.min())}**")
    else:
        st.info("Not enough history to compute drawdown.")

    st.subheader("Methodology & Assumptions")
    st.markdown("""
- **Fair Price (Band):** sector percentiles (P25/P50/P75) across EV/EBITDA, EV/Sales, and P/E; net debt & cash applied.
- **Scenarios (1Y):** Monte Carlo with EWMA (λ = 0.94). Representative paths by terminal quantiles (P10/P50/P90).
- **Financials:** we hide lines that are all-NaN, all-zero, or have <50% data coverage.
- **Dividends (TTM):** cash paid over the last 365 days; payout = Div / (FCF or last FY Net Income).
- **Risk:** historical Vol/ VaR; 90-day rolling correlation/beta; 5Y drawdown.
    """)

with tabs[7]:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("### Notes")
        st.info(
            "- Universe file: `data/companies_list.csv` (ticker, name, index, sector)\n"
            "- Prices & fundamentals: Yahoo Finance via `yfinance`\n"
            "- Scenario chart shows 5Y history and 1Y MC EWMA quantiles (Bear/Base/Bull)."
        )
        with st.expander("Data caveats", expanded=False):
            st.markdown(
                "- Some fundamentals can be missing across fiscal years; we hide sparse lines.\n"
                "- EV/EBITDA can be undefined when EBITDA ≤ 0.\n"
                "- Dividends use cash paid (TTM); special dividends may distort payout.\n"
                "- MC-EWMA scenarios assume stationary μ (daily mean) and current σ level; guidance, not forecasts."
            )

        st.markdown("### Session")
        st.caption(
            f"Last refresh: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  •  "
            f"Index map: S&P 500→^GSPC, IPC→^MXX, Ibovespa→^BVSP"
        )

    with right:
        st.markdown("### Downloads")
        with st.spinner("Preparing CSVs…"):
            inc, bal, cfs = get_statements(ticker, last_years=4)
            r1, r2 = st.columns(2)
            with r1:
                st.download_button("Peers (CSV)", pd.DataFrame(cmp_df).to_csv(index=True).encode("utf-8"),
                                   file_name=f"{ticker}_peers.csv", use_container_width=True)
            with r2:
                st.download_button("Income (CSV)", inc.to_csv().encode("utf-8"),
                                   file_name=f"{ticker}_income.csv", use_container_width=True)
            r3, r4 = st.columns(2)
            with r3:
                st.download_button("Balance (CSV)", bal.to_csv().encode("utf-8"),
                                   file_name=f"{ticker}_balance.csv", use_container_width=True)
            with r4:
                st.download_button("Cash Flow (CSV)", cfs.to_csv().encode("utf-8"),
                                   file_name=f"{ticker}_cashflow.csv", use_container_width=True)
