# Company Valuation

[![Launch App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://company-valuation.streamlit.app/)

From a single ticker to a decision-ready view of value.  
This project delivers a focused, repeatable **single-company valuation workflow** that blends live market data with clean fundamentals, peer context, and scenario analysis—packaged in a simple, audience-friendly Streamlit app.

---

## Quick Access
- Dashboard: [Streamlit App](https://company-valuation.streamlit.app)  
- Notebook: [`notebook.ipynb`](notebook.ipynb)

---

## Highlights
- **Smart search (type-ahead)** over a curated universe (S&P 500, IPC, Ibovespa) with Yahoo-formatted tickers.  
- **Overview**: 1M/3M/YTD/1Y returns, 1Y max drawdown, profile, and an inline **Buy/Hold/Sell** label.  
- **Price & Benchmark**: 5Y performance vs sector & local index, plus **Sharpe** and **beta**.  
- **Peers & Sector**: Comparable **EV/EBITDA, EV/Sales, P/E** with sector medians and percentile chips.  
- **Financials (last 4 FY)**: Income, Balance, Cash Flow with **auto-hidden** sparse/irrelevant lines (e.g., Inventory for banks) and headline ratios (Margins, ROE/ROA/ROIC, Liquidity, Net Debt/EBITDA).  
- **Fair Price**: Peer-calibrated valuation band (P25/P50/P75) → **implied upside** and **recommendation**.  
- **Scenarios (1Y)**: Monte Carlo paths on top of **5Y history** with **EWMA volatility** (bear/base/bull).  
- **Dividends**: Last 4 years by month, **Dividend Yield (TTM)**, peer median, and **Payout (DPS/EPS TTM)**.  
- **Advanced**: Risk kit (vol, 1-day VaR), rolling correlation/beta, and 5Y drawdown.  
- **Downloads**: One-click CSVs for peers and statements.

---

## Repository Structure
- `app.py` — Streamlit application.  
- `notebook.ipynb` — Storytelling notebook (clean pipeline + spot checks).  
- `data/companies_list.csv` — Universe with columns: `ticker`, `name`, `index`, `sector`.  
- `requirements.txt` — Dependencies.  
- `README.md` — This file.

---

## Data
- **Source**: Yahoo Finance via `yfinance` (live at runtime; no API keys).  
- **Universe**: S&P 500 (U.S.), S&P/BMV IPC (Mexico), Ibovespa (Brazil).  
  - Tickers must be Yahoo-compatible (e.g., Mexican tickers end with `.MX`, Brazilian with `.SA`).  
- **Notes**: Fundamentals can be incomplete/lagged for some names. The app hides all-NaN/all-zero lines and suppresses non-meaningful multiples (e.g., EBITDA ≤ 0).

---

## Methodology (Short)
- **Peers**: Same **index** and **sector** as the selected company.  
- **Fair Price**: For each multiple (EV/EBITDA, EV/Sales, P/E), apply sector **P25/P50/P75** to company bases, adjust for **net debt/cash**, divide by shares; aggregate across methods for Low/Mid/High.  
- **Recommendation**: `BUY` (upside ≥ +15%), `SELL` (≤ –15%), else `HOLD`.  
- **Scenarios**: 1-year Monte Carlo with **EWMA (λ=0.94)** volatility; paths aligned to bear/base/bull quantiles.  
- **Payout**: Dividend Yield (TTM) from trailing dividends; **Payout = DPS (TTM) / EPS (TTM)** with fallbacks if EPS TTM unavailable.

---

## Run Locally
```bash
# 1) Clone
git clone https://github.com/<your-user>/Company-Valuation.git
cd Company-Valuation

# 2) (Optional) venv
python3 -m venv .venv && source .venv/bin/activate

# 3) Install
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

# 4) Launch
python3 -m streamlit run app.py
