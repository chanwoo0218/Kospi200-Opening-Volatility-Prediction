"""FRED에서 미국 10년물 국채금리를 수집합니다."""
import os
from pathlib import Path
import pandas as pd
from fredapi import Fred

api_key = os.environ.get("FRED_API_KEY")
if not api_key:
    raise RuntimeError("FRED_API_KEY 환경변수를 설정하세요.")

fred = Fred(api_key=api_key)
us10y = fred.get_series("DGS10", observation_start="2021-01-01")
df = pd.DataFrame(us10y, columns=["US10Y"])
df.index.name = "Date"
output = Path("data/processed/us10y_treasury_5y.csv")
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, encoding="utf-8-sig")
print(f"저장 완료: {output} ({len(df):,}일)")
