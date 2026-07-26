# 야간 글로벌 금융지표 기반 KOSPI 200 개장 변동성 예측

> 미국 증시·VIX·환율·미국 10년물 금리와 전일 국내시장 정보를 활용해 KOSPI 200의 **시초가 갭 충격(Y1)**과 **개장 후 30분 실현변동성(Y2)**을 예측하고 SHAP으로 근거를 해석한 시계열 머신러닝 프로젝트입니다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 수행 기간 | 2026.03.13 ~ 2026.05.10 |
| 문제 유형 | 금융 시계열 회귀·설명가능 AI |
| 관측 단위 | 거래일 |
| 목표값 | Y1 시초가 갭 충격, Y2 09:00~09:30 변동성 |
| 최종 모델 | 튜닝된 XGBoost |
| 검증 | 시간순 80:20 분할 + TimeSeriesSplit |

## 발표자료

[![PDF 발표자료](https://img.shields.io/badge/PDF-원본_발표자료_보기-EA4335?logo=adobeacrobatreader&logoColor=white)](docs/산업경영%20발표자료.pdf)

- [원본 발표자료 PDF 열기](docs/산업경영%20발표자료.pdf)
- [발표자료 핵심 내용 보기](docs/발표자료_요약.md)

## 핵심 설계

- 미국 시장 D일 종가 정보를 한국 D+1일 개장에 맞춰 정렬
- 예측 시점인 07:59 이전에 확정된 정보만 사용
- 국내 당일 변수는 직접 사용하지 않고 1~3일 lag 적용
- 이동통계는 `shift(1)` 후 계산하여 당일 정보 누수 방지
- 무작위 분할 대신 시간순 분할과 expanding-window 교차검증 사용
- OLS, Random Forest, XGBoost 성능 비교 후 TreeSHAP 해석

## 주요 결과

### Y1: 시초가 갭 충격

| 모델 | 테스트 MAE | 테스트 RMSE |
|---|---:|---:|
| OLS | 0.01803 | 0.02455 |
| Random Forest | 0.01771 | 0.02516 |
| **XGBoost** | **0.01714** | **0.02382** |

### Y2: 개장 후 30분 변동성

| 모델 | 테스트 MAE | 테스트 RMSE |
|---|---:|---:|
| OLS | 0.00492 | 0.00640 |
| Random Forest | 0.00487 | 0.00624 |
| **XGBoost** | **0.00481** | **0.00619** |

## 저장소 구성

```text
.
├── README.md
├── requirements.txt
├── data/processed
│   ├── Vol_Pred_Final_KST_Data_v2.csv
│   └── us10y_treasury_5y.csv
├── docs
│   ├── 산업경영 발표자료.pdf
│   └── 발표자료_요약.md
└── src
    ├── collect_market_data.py
    ├── collect_us10y.py
    ├── modeling_predict_y1.py
    └── modeling_predict_y2.py
```

## 실행 방법

```bash
pip install -r requirements.txt
python scripts/reassemble_data.py
python src/modeling_predict_y1.py
python src/modeling_predict_y2.py
```

새로운 미국 10년물 금리 데이터를 수집하려면 FRED API 키를 코드에 입력하지 말고 환경변수로 설정합니다.

```bash
export FRED_API_KEY="발급받은_키"
python src/collect_us10y.py
```

## 보안 안내

원본 ZIP의 테스트 파일에서 하드코딩된 API 인증정보가 발견되어 해당 파일은 업로드하지 않았습니다. 노출된 키는 폐기·재발급한 뒤 환경변수 방식으로 사용해야 합니다.

## 한계

- 예측값은 통계적 위험지표이며 거래수익을 보장하지 않습니다.
- 거래비용·슬리피지·실행 가능한 투자규칙은 평가하지 않았습니다.
- 극단적 변동 구간은 표본이 적어 성능이 불안정할 수 있습니다.

> 본 저장소는 연구·교육 목적이며 투자 조언이 아닙니다.