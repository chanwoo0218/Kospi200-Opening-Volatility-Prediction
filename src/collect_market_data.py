from pathlib import Path
Path("data/processed").mkdir(parents=True, exist_ok=True)
"""
FinanceDataReader로 국내·해외 금융변수를 수집하고 KOSPI 200 개장 변동성 분석용 변수를 생성합니다.
"""

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
from time import sleep

START_DATE = "2022-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

print("=" * 60)
print("전체 변수 수집 시작 (FinanceDataReader)")
print(f"기간: {START_DATE} ~ {END_DATE}")
print("=" * 60)

kospi200 = fdr.DataReader("KS200", START_DATE, END_DATE)[["Open", "High", "Low", "Close", "Volume"]]
sleep(1)

# 선물 직접 조회가 어려워 KODEX 200 ETF를 프록시로 사용
kodex200 = fdr.DataReader("069500", START_DATE, END_DATE)[["Open", "High", "Low", "Close", "Volume"]]
kodex200.columns = ["FUT_Open", "FUT_High", "FUT_Low", "FUT_Close", "FUT_Volume"]
sleep(1)

sp500 = fdr.DataReader("US500", START_DATE, END_DATE)[["Close"]]
sp500.columns = ["SP500_Close"]
sleep(1)
nasdaq = fdr.DataReader("IXIC", START_DATE, END_DATE)[["Close"]]
nasdaq.columns = ["NASDAQ_Close"]
sleep(1)
vix = fdr.DataReader("VIX", START_DATE, END_DATE)[["Close"]]
vix.columns = ["VIX_Close"]
sleep(1)
usdkrw = fdr.DataReader("USD/KRW", START_DATE, END_DATE)[["Close"]]
usdkrw.columns = ["USDKRW_Close"]

domestic = kospi200.join(kodex200, how="inner")
overseas = sp500.join(nasdaq, how="outer").join(vix, how="outer").join(usdkrw, how="outer")
# 미국 D일 종가를 한국 D+1일 개장 정보에 정렬
overseas.index = overseas.index + timedelta(days=1)
df = domestic.join(overseas, how="left")
global_cols = ["SP500_Close", "NASDAQ_Close", "VIX_Close", "USDKRW_Close"]
df[global_cols] = df[global_cols].ffill()
df = df.dropna()

df["x1_kospi200_return"] = df["Close"].pct_change()
df["x2_kospi200_range"] = (df["High"] - df["Low"]) / df["Close"]
df["x3_kospi200_abs_oc"] = (df["Open"] - df["Close"]).abs() / df["Close"]
df["x4_futures_return"] = df["FUT_Close"].pct_change()
df["x5_futures_range"] = (df["FUT_High"] - df["FUT_Low"]) / df["FUT_Close"]
df["x6_basis"] = df["Close"].pct_change() - df["FUT_Close"].pct_change()
df["x7_sp500_return"] = df["SP500_Close"].pct_change()
df["x8_nasdaq_return"] = df["NASDAQ_Close"].pct_change()
df["x9_vix_change"] = df["VIX_Close"].pct_change()
df["x10_usdkrw_change"] = df["USDKRW_Close"].pct_change()
df["Y1_shock_vol"] = (df["Open"] - df["Close"].shift(1)).abs() / df["Close"].shift(1)
df = df.dropna()

variable_cols = [
    "x1_kospi200_return", "x2_kospi200_range", "x3_kospi200_abs_oc",
    "x4_futures_return", "x5_futures_range", "x6_basis",
    "x7_sp500_return", "x8_nasdaq_return", "x9_vix_change",
    "x10_usdkrw_change", "Y1_shock_vol",
]

df.to_csv("data/processed/kospi200_all_data.csv", encoding="utf-8-sig")
df[variable_cols].to_csv("data/processed/kospi200_model_variables.csv", encoding="utf-8-sig")
print(f"수집 완료: {len(df):,}일")
