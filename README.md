# KOSPI 200 Opening Volatility Prediction

> **야간 글로벌 금융시장 정보와 전일 국내시장 데이터를 활용해 KOSPI 200의 시초가 갭 충격과 개장 후 30분 변동성을 예측하고, SHAP으로 예측 근거를 해석한 시계열 머신러닝 프로젝트입니다.**

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![SHAP](https://img.shields.io/badge/XAI-TreeSHAP-green)

## At a Glance

| Item | Description |
|---|---|
| Project type | Financial time-series regression / Explainable AI |
| Period | 2026.03.13 - 2026.05.10 |
| Activity | DF regular project |
| Observation unit | Trading day |
| Raw analysis period | 2021-05-05 - 2026-03-31 |
| Targets | Y1 opening-gap shock, Y2 09:00-09:30 realized volatility |
| Final model | Tuned XGBoost |
| Validation | Chronological 80:20 holdout + `TimeSeriesSplit` |
| Core stack | Python, Pandas, scikit-learn, XGBoost, SHAP |

## Problem

The Korean stock market absorbs information generated after its previous close, including movements in the US equity market, volatility index, exchange rate, and interest rates. This information is often reflected intensively during the opening period of the next Korean trading day.

This project asks:

> **How much of the KOSPI 200 opening shock and early-session volatility can be predicted using information that is actually available before the market opens?**

Rather than predicting only market direction, the project focuses on the **magnitude of opening risk** and explains which variables drive each prediction.

## Targets

- **Y1 - Opening-gap shock:** magnitude of the gap between the previous close and the current opening price
- **Y2 - Opening realized volatility:** realized volatility during the 09:00-09:30 KST interval

## Dataset

The integrated dataset contains **1,280 trading-day observations and 23 source columns** before additional lag and rolling features are generated.

| Category | Variables |
|---|---|
| Korean market | KOSPI 200 OHLC, returns, volume, intraday absolute return, volatility proxy |
| US equity market | S&P 500 return, NASDAQ return |
| Risk and macro | VIX change, USD/KRW return, US 10-year Treasury yield change |
| Engineered history | Domestic lags 1-3, global lags 1-2, five-day rolling statistics |

US market data for day `D` is shifted to the Korean trading day `D+1` to reflect when that information becomes available to the domestic market.

## Leakage Prevention

Financial prediction is easily overstated when the model uses values unavailable at prediction time. The pipeline therefore applies a **07:59 KST cutoff rule**.

- Only global variables confirmed before the Korean open are used at time `t`.
- Same-day Korean market variables are never used directly.
- Domestic market variables are shifted by 1-3 trading days.
- Rolling features are calculated after `shift(1)` so the current day is excluded.
- Data is split chronologically instead of randomly.
- Hyperparameter tuning uses expanding-window `TimeSeriesSplit`.

## Feature Engineering

The final modeling matrix uses **28 predictors**:

- Five overnight global variables
- Eight lagged global-market variables
- Twelve lagged domestic-market variables
- Three five-day rolling statistics

These features are designed to represent both immediate overnight shocks and volatility-clustering effects.

## Modeling Pipeline

```text
Market data collection
        ↓
KST trading-date alignment
        ↓
Lag and rolling feature engineering
        ↓
07:59 cutoff / leakage audit
        ↓
Chronological train-test split
        ↓
OLS → Random Forest → XGBoost
        ↓
MAE / RMSE comparison
        ↓
Global and local TreeSHAP interpretation
```

## Results

### Y1 - Opening-gap shock

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| OLS | 0.01803 | 0.02455 |
| Random Forest | 0.01771 | 0.02516 |
| **XGBoost** | **0.01714** | **0.02382** |

### Y2 - Opening realized volatility

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| OLS | 0.00492 | 0.00640 |
| Random Forest | 0.00487 | 0.00624 |
| **XGBoost** | **0.00481** | **0.00619** |

For Y2, the tuned XGBoost model reduced MAE by approximately **21%** relative to the documented naive baseline MAE of `0.00607`.

## Model Interpretation

TreeSHAP was used at two levels.

- **Global explanation:** identifies variables that consistently influence predictions across the test period.
- **Local explanation:** explains which features caused an extreme day's prediction to increase or decrease.

Key findings from the submitted analysis:

- Y1 was strongly affected by recent KOSPI returns, trading volume, and the five-day KOSPI return trend.
- Y2 was strongly affected by lagged realized volatility and its recent five-day average.
- Extreme cases were inspected separately rather than relying only on aggregate feature importance.

## My Contribution

- Collected and integrated KOSPI 200, KODEX 200, S&P 500, NASDAQ, VIX, USD/KRW, and US Treasury data.
- Designed the `D → D+1` alignment logic between the US close and the next Korean open.
- Built lag and rolling features reflecting recent return and volatility dynamics.
- Designed the 07:59 cutoff rule and audited the pipeline for temporal leakage.
- Implemented chronological validation and `TimeSeriesSplit` tuning.
- Compared OLS, Random Forest, and XGBoost and interpreted the final model with TreeSHAP.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data
│   ├── README.md
│   └── sample
│       └── kospi200_model_variables_sample.csv
├── docs
│   └── presentation_summary.md
└── src
    ├── collect_data.py
    ├── fetch_us10y.py
    ├── modeling.py
    ├── modeling_predict_y1.py
    └── modeling_predict_y2.py
```

## How to Run

```bash
git clone https://github.com/chanwoo0218/Kospi200-Opening-Volatility-Prediction.git
cd Kospi200-Opening-Volatility-Prediction
pip install -r requirements.txt
```

Collect public market data:

```bash
python src/collect_data.py
```

The US Treasury collector requires a FRED API key. The real key is never stored in the repository.

```bash
export FRED_API_KEY="your_api_key"
python src/fetch_us10y.py --output data/raw/us10y_treasury_5y.csv
```

Run the two target models after placing the processed dataset in the expected path:

```bash
python src/modeling_predict_y1.py
python src/modeling_predict_y2.py
```

## Limitations

- The model predicts statistical risk measures, not trading profitability.
- Transaction costs, slippage, and executable investment rules were not evaluated.
- Market regimes may change, so historical performance does not guarantee future performance.
- Extreme-event performance remains less stable because such observations are rare.

## Future Work

- Walk-forward backtesting across multiple market regimes
- Prediction intervals and uncertainty calibration
- Regime-aware or volatility-state models
- Real-time ingestion and pre-open inference pipeline
- Evaluation using decision-oriented risk metrics

## Portfolio

A Korean-language project narrative, responsibilities, and learning reflections are available on the [Notion portfolio page](https://app.notion.com/p/81e82d8994c28379910601cc7c17706c).

> This repository is for research and educational purposes and does not constitute investment advice.