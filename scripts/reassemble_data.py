"""분할된 최종 모델링 데이터를 원래 파일로 복원합니다."""
from pathlib import Path
import pandas as pd

base = Path("data/processed")
parts = sorted(base.glob("Vol_Pred_Final_KST_Data_v2_part*.csv"))
if not parts:
    raise FileNotFoundError("Vol_Pred_Final_KST_Data_v2_part*.csv가 없습니다.")
pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(base / "Vol_Pred_Final_KST_Data_v2.csv", index=False)
print(f"복원 완료: {base / 'Vol_Pred_Final_KST_Data_v2.csv'}")
