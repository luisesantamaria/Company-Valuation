import streamlit as st, pandas as pd, numpy as np
st.set_page_config(page_title="Smoke", layout="wide")
st.title("It lives! ✅")
st.write(pd.DataFrame({"x": range(10), "y": np.random.randn(10)}))
