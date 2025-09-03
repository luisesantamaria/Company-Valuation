import streamlit as st, pandas as pd, numpy as np
st.set_page_config(page_title="Step 1", layout="wide")
st.title("Step 1 — pandas & numpy OK")

st.write("Random table")
df = pd.DataFrame({"x": range(10), "y": np.random.randn(10)})
st.dataframe(df, use_container_width=True)
