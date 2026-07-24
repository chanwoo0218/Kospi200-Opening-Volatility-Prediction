# 데이터 안내

`processed/`에는 실제 최종 모델 코드가 읽는 통합 데이터와 미국 10년물 금리 데이터를 배치합니다.

- `Vol_Pred_Final_KST_Data_v2.csv`: Y1·Y2 모델링에 사용한 최종 통합 데이터
- `us10y_treasury_5y.csv`: 미국 10년물 국채금리

ZIP에 포함된 `Vol_Pred_Final_KST_Data.csv`, `kospi200_all_data.csv`, `kospi200_model_variables.csv`는 이전·중간 산출물입니다. `src/collect_market_data.py`를 실행하면 기본 시장 데이터와 변수표를 다시 생성할 수 있습니다.

현재 GitHub 연결에서는 로컬 CSV 파일을 직접 전달하는 기능이 없어 최종 통합 CSV는 별도 업로드가 필요합니다. 저장소 루트에서 `data/processed/Vol_Pred_Final_KST_Data_v2.csv` 경로로 배치하면 모델 코드를 실행할 수 있습니다.
