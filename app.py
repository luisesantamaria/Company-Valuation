import sys
import numpy as np
import pandas as pd
import yfinance as yf
import altair as alt
import streamlit as st

st.set_page_config(page_title="Company Valuation — Smoke Test", layout="wide")
st.title("Company Valuation — Smoke Test")

# --- UI superior: ticker + versiones ---
left, right = st.columns([2, 1])
with left:
    ticker = st.text_input("Ticker (Yahoo Finance)", "AAPL").strip().upper()
with right:
    st.caption("Environment")
    st.write(
        f"streamlit **{st.__version__}** · pandas **{pd.__version__}** · "
        f"numpy **{np.__version__}** · yfinance **{yf.__version__}** · "
        f"python **{sys.version.split()[0]}**"
    )

@st.cache_data(ttl=900, show_spinner=False)
def get_hist(tk: str, period="1y") -> pd.Series:
    try:
        df = yf.download(tk, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df:
            s = df["Close"].dropna()
            s.name = tk
            return s
    except Exception:
        pass
    return pd.Series(dtype=float)

# --- Descarga y salida temprana si falla ---
s = get_hist(ticker)
if s.empty:
    st.warning("No pudimos descargar datos para ese ticker. Prueba con AAPL, MSFT, AMZN, etc.")
    st.stop()

# --- Métricas rápidas ---
def pct_since(n_days: int) -> float:
    if len(s) > n_days:
        return float(s.iloc[-1] / s.iloc[-(n_days + 1)] - 1.0)
    return np.nan

# YTD: desde el 1 de enero
try:
    year_start = pd.Timestamp(pd.Timestamp.utcnow().year, 1, 1)
    idx = np.searchsorted(s.index.values, np.datetime64(year_start))
    ytd = float(s.iloc[-1] / s.iloc[idx] - 1.0) if idx < len(s) else np.nan
except Exception:
    ytd = np.nan

r_1m = pct_since(21)
r_3m = pct_since(63)
r_1y = float(s.iloc[-1] / s.iloc[0] - 1.0)

cols = st.columns(5)
labels = ["1M", "3M", "YTD", "1Y", "Last Price"]
vals   = [r_1m, r_3m, ytd, r_1y, float(s.iloc[-1])]
fmts   = [lambda v: f"{v*100:.1f}%", lambda v: f"{v*100:.1f}%", lambda v: f"{v*100:.1f}%", lambda v: f"{v*100:.1f}%", lambda v: f"{v:,.2f}"]

for c, (lab, val, fmt) in zip(cols, zip(labels, vals, fmts)):
    if np.isfinite(val):
        c.metric(lab, fmt(val))

# --- Gráfica: 1Y Price ---
df = s.reset_index().rename(columns={"index": "Date", s.name: "Price"})
chart = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Price:Q", title="Price", scale=alt.Scale(zero=False, nice=True)),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Price:Q", format=",.2f")],
    )
    .properties(height=320, title=f"{ticker} — 1Y Price")
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

st.caption("Si ves la tabla de métricas y la línea de precio, el entorno y las dependencias están OK.")
