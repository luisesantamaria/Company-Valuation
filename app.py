# app.py — Smoke Test for Streamlit Cloud

import platform
import traceback
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Smoke Test • Company Valuation", layout="wide")

def main():
    st.title("🚦 Streamlit Cloud — Smoke Test")
    st.caption("If you can see this page (with charts and tables), your environment is OK.")

    # --- Versions ------------------------------------------------------------
    import yfinance as yf
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Python", platform.python_version())
    c2.metric("streamlit", st.__version__)
    c3.metric("pandas", pd.__version__)
    c4.metric("numpy", np.__version__)
    c5.metric("altair", alt.__version__)
    c6.metric("yfinance", yf.__version__)

    st.divider()

    # --- Filesystem check ----------------------------------------------------
    st.subheader("Filesystem check")
    csv_path = Path(__file__).parent / "data" / "companies_list.csv"
    if csv_path.exists():
        st.success(f"Found: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            st.write("Top rows from `companies_list.csv`:")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error("Could not read `companies_list.csv`.")
            st.exception(e)
    else:
        st.warning("Missing `data/companies_list.csv` in the repo root.")

    st.divider()

    # --- Local chart (no network) -------------------------------------------
    st.subheader("Altair chart (local data)")
    x = np.arange(60)
    y = np.cumsum(np.random.randn(60))
    local_df = pd.DataFrame({"Date": pd.date_range(end=datetime.utcnow(), periods=60), "Series": y})
    ch = (
        alt.Chart(local_df)
        .mark_line()
        .encode(x=alt.X("Date:T", title=None), y=alt.Y("Series:Q", title="Random Walk"))
        .properties(height=240)
        .interactive()
    )
    st.altair_chart(ch, use_container_width=True)

    st.divider()

    # --- Network test: tiny yfinance fetch ----------------------------------
    st.subheader("Network test (yfinance: AAPL, 1 month)")
    try:
        with st.spinner("Downloading AAPL (1mo)…"):
            df_px = yf.download("AAPL", period="1mo", interval="1d", auto_adjust=True, progress=False)
        if df_px is None or df_px.empty:
            st.warning("No data returned by yfinance (AAPL, 1mo).")
        else:
            price = df_px["Close"].rename("Price").reset_index()
            ch2 = (
                alt.Chart(price)
                .mark_line()
                .encode(
                    x=alt.X(price.columns[0] + ":T", title=None),
                    y=alt.Y("Price:Q", title="AAPL Price", scale=alt.Scale(zero=False, nice=True)),
                    tooltip=[alt.Tooltip(price.columns[0] + ":T"), alt.Tooltip("Price:Q", format=",.2f")]
                )
                .properties(height=240)
                .interactive()
            )
            st.altair_chart(ch2, use_container_width=True)
            st.success("yfinance fetch OK.")
    except Exception as e:
        st.error("yfinance raised an exception:")
        st.exception(e)

    st.caption("End of smoke test.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("The app crashed during startup. Details below:")
        st.exception(e)
        st.code(traceback.format_exc())
