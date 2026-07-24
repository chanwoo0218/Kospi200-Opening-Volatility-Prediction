# KOSPI 200 Opening Volatility Prediction

야간 글로벌 금융시장 정보와 전일 국내시장 정보를 활용하여 **KOSPI 200 개장 초기 가격 충격과 변동성**을 예측하고, SHAP으로 주요 영향 변수를 해석한 프로젝트입니다.

## Problem

미국 증시, NASDAQ, VIX, 원/달러 환율 등 야간 정보가 다음 거래일 국내 시장 개장 초기에 어떻게 반영되는지 분석합니다. 데이터 누수를 막기 위해 예측 시점 이전에 확정된 변수만 사용하고, 시간 순서를 유지한 검증을 적용합니다.

## Workflow

1. FinanceDataReader로 KOSPI 200, KODEX 200, S&P 500, NASDAQ, VIX, USD/KRW 수집
2. 미국 D일 종가를 한국 D+1일 개장과 정렬
3. 수익률·고저폭·베이시스·변동성 파생변수 생성
4. 시간순 80:20 분할 및 TimeSeriesSplit
5. OLS, Random Forest, XGBoost 비교
6. 최종 트리 모델을 SHAP으로 해석

## Repository Structure

```text
src/collect_data.py   # public market-data collection and alignment
src/modeling.py       # time-aware model comparison and SHAP export
data/                 # generated or externally supplied data
notebooks/            # original notebook can be added here
```

## Run

```bash
pip install -r requirements.txt
python src/collect_data.py
python src/modeling.py --data kospi200_model_variables.csv
```

## Notes

- 공개용 저장소를 위해 데이터 수집과 모델링 흐름을 스크립트로 정리했습니다.
- 투자 조언을 목적으로 하지 않으며, 연구·교육 목적의 분석입니다.
