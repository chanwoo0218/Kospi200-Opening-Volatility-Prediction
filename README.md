# KOSPI 200 Opening Volatility Prediction

미국 시장 마감 이후 한국 시장 개장 전에 확정되는 글로벌 금융정보와 전일 국내시장 정보를 이용해 **KOSPI 200 개장 충격(Y1)**과 **09:00–09:30 실현변동성(Y2)**을 예측한 시계열 회귀 프로젝트입니다.

## Targets

- **Y1 — Opening gap shock:** 당일 시가와 전일 종가 사이의 절대 갭
- **Y2 — Opening realized volatility:** 개장 후 30분 구간의 로그수익률 기반 실현변동성

## Leakage Control

- 미국 D일 종가는 한국 D+1일 개장 정보로 날짜를 +1일 정렬
- 모든 국내 변수는 lag 1–3 또는 과거 rolling statistics로 변환
- 예측 기준시각을 **07:59 KST**로 두고 이후에 확정되는 값은 배제
- 무작위 분할 대신 시간순 80:20 holdout과 `TimeSeriesSplit` 사용

## Features

최종 모델에는 28개 변수를 사용했습니다.

- 당일 새벽 확정: S&P 500, NASDAQ, VIX, USD/KRW, 미국 10년물 금리 변화
- 글로벌 변수 lag 1–2
- KOSPI 수익률, 거래량, 변동성 proxy, 장중 절대수익률 lag 1–3
- 과거 5일 이동평균·표준편차

## Results

### Y1: Opening gap shock

| Model | MAE | RMSE |
|---|---:|---:|
| OLS | 0.01803 | 0.02455 |
| Random Forest | 0.01771 | 0.02516 |
| XGBoost | **0.01714** | **0.02382** |

### Y2: 09:00–09:30 realized volatility

| Model | MAE | RMSE |
|---|---:|---:|
| OLS | 0.00492 | 0.00640 |
| Random Forest | 0.00487 | 0.00624 |
| XGBoost | **0.00481** | **0.00619** |

Y2의 naive baseline MAE 0.00607 대비 XGBoost는 약 **21%**의 오차 감소를 보였습니다.

## Interpretation

TreeSHAP 분석에서:

- Y1은 전일 KOSPI 수익률·거래량과 KOSPI 5일 평균의 영향이 컸습니다.
- Y2는 전일 실현변동성과 5일 변동성 평균이 핵심이었습니다.

극단 사례로 Y1은 2026-02-05(`y1_shock=0.0954`), Y2는 2025-09-15(`log_target_y2=0.0310`)를 로컬 SHAP으로 분석했습니다.

## Repository Structure

```text
src/collect_data.py
src/fetch_us10y.py
src/modeling_predict_y1.py
src/modeling_predict_y2.py
data/README.md
docs/presentation_summary.md
```

## Run

```bash
pip install -r requirements.txt
python src/collect_data.py --output-dir data/processed
export FRED_API_KEY="your_key"
python src/fetch_us10y.py --output data/raw/us10y_treasury_5y.csv
python src/modeling_predict_y1.py --data data/processed/Vol_Pred_Final_KST_Data_v2.csv
python src/modeling_predict_y2.py --data data/processed/Vol_Pred_Final_KST_Data_v2.csv
```

공개 저장소에는 API 키를 포함하지 않습니다. 첨부된 실험용 `test.py`에 있던 하드코딩 키는 제거하고 환경변수 방식으로 대체했습니다. 이 저장소는 연구·교육 목적이며 투자 조언이 아닙니다.