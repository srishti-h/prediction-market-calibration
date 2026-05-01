# Prediction Market Calibration Study

An empirical analysis of calibration, bias, and efficiency in binary prediction markets (Polymarket + Kalshi), examining whether market-implied probabilities accurately reflect real-world outcomes.

## Key Findings

| Metric | Value |
|--------|-------|
| Markets analyzed | 3,960 resolved binary markets, 108K+ price records |
| Overall ECE (T-24h) | 0.095 |
| Brier Score | 0.193 |
| Logistic slope | 1.33 (underconfident — prices too moderate) |
| Crypto ECE | 0.052 (most efficient) |
| Sports ECE | 0.260 (least efficient) |
| Favorite bias | +12.6pp (actual 90.7% vs predicted 78.2%, p=0.0015) |
| Contested bias | −6.7pp (actual 42.9% vs predicted 49.6%, p=0.0008) |
| AUC at T-168h | 0.644 → T-72h 0.892 (price discovery window) |
| Liquidity-calibration r | −0.172 (p≈0) — higher volume = better calibrated |
| Isotonic recalibration ECE | 0.049 (−44% vs uncalibrated, 5-fold CV) |
| Backtest ROI (quarter-Kelly, OOS) | +703% over 732 bets, Sharpe 5.30, max DD 7.1% |

## Research Questions

1. **Calibration**: When a market prices an event at 70%, does it resolve YES ~70% of the time?
2. **Favorite-Longshot Bias**: Do markets systematically overestimate low-probability events?
3. **Category Efficiency**: Are sports markets more efficiently priced than political or crypto markets?
4. **Temporal Efficiency**: Does market calibration improve as the resolution date approaches?
5. **Liquidity Effect**: Do higher-volume markets exhibit less mispricing?

## Project Structure

```
prediction-market-calibration/
├── src/
│   ├── scrapers/
│   │   ├── polymarket.py     # Gamma API + CLOB API scraper
│   │   └── kalshi.py         # Kalshi v2 API scraper
│   ├── pipeline/
│   │   ├── ingest.py         # DuckDB schema + bulk inserts
│   │   └── features.py       # Feature engineering
│   ├── analysis/
│   │   ├── calibration.py    # ECE, Brier score, reliability diagrams
│   │   ├── bias.py           # Favorite-longshot, overconfidence tests
│   │   └── efficiency.py     # Price drift, AUC by horizon, category comparison
│   └── viz/
│       └── plots.py          # Matplotlib publication-quality figures
├── scripts/
│   ├── collect_data.py       # Entry point: scrape + store
│   └── run_analysis.py       # Entry point: analyze + generate figures
├── dashboard/
│   └── app.py                # Streamlit interactive dashboard
└── data/                     # DuckDB database (gitignored)
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/prediction-market-calibration
cd prediction-market-calibration
pip install -r requirements.txt
cp .env.example .env        # optional: add Kalshi credentials
```

## Usage

### Step 1: Collect data

```bash
# Scrape ~2000 resolved Polymarket markets (~15-20 min)
python scripts/collect_data.py --markets 2000

# Also scrape Kalshi (requires .env credentials)
python scripts/collect_data.py --markets 2000 --kalshi
```

### Step 2: Run analysis

```bash
python scripts/run_analysis.py
```

Figures are saved to `outputs/figures/`. Summary stats to `outputs/summary.json`.

### Step 3: Interactive dashboard

```bash
streamlit run dashboard/app.py
```

## Analysis Methods

### Calibration
- **Expected Calibration Error (ECE)**: Weighted average of `|predicted - actual|` per probability bin. Perfect calibration = 0.
- **Brier Score**: Mean squared error between predicted probability and binary outcome.
- **Reliability Diagram**: Visual comparison of predicted vs. actual resolution rates per decile bin.

### Bias
- **Favorite-Longshot Bias**: Compare actual resolution rates for longshots (<30%), contested (30-70%), and favorites (>70%). Classic finding: longshots overpriced, favorites underpriced.
- **Logistic Calibration Curve**: Fit `P(YES) ~ logistic(β₀ + β₁ · log_odds(implied))`. Slope < 1 = overconfidence.
- **Volume-Bias Correlation**: Does higher liquidity predict better calibration?

### Efficiency
- **AUC by Horizon**: Binary classifier AUC for YES resolution using price at T-168h, T-72h, T-24h, T-1h. Rising AUC = information incorporation.
- **Price Drift**: Average price trajectory for YES vs. NO resolving markets. Divergence near close = market learning.
- **Category Comparison**: ECE by market type (sports, politics, crypto, economics).

## Data Sources

- **Polymarket**: Public Gamma API (`gamma-api.polymarket.com`) and CLOB API. No auth required.
- **Kalshi**: Public trading API v2 (`trading-api.kalshi.com`). Some endpoints require free account.

## Resume Bullets

```
Prediction Market Calibration Study | Python, DuckDB, Polymarket API
- Scraped and analyzed [N]+ resolved binary markets across 5+ categories
  using Polymarket's public REST + CLOB APIs
- Quantified favorite-longshot bias: longshot markets (<30% implied) resolved
  YES at Xx their implied rate (ECE = X.XXX, p < 0.05)
- Demonstrated category-level efficiency gap between sports (ECE=X.XXX) and
  crypto markets (ECE=X.XXX), consistent with informed-trader hypothesis
- Built Streamlit dashboard with interactive reliability diagrams, temporal
  calibration charts, and price drift analysis across [N]+ price snapshots
```
