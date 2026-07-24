from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default="DGS10")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/us10y_treasury_5y.csv"))
    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Set FRED_API_KEY in the environment before running this script.")

    response = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": args.series,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": args.start,
            "observation_end": args.end,
        },
        timeout=30,
    )
    response.raise_for_status()
    observations = response.json()["observations"]
    frame = pd.DataFrame(observations)[["date", "value"]]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna().rename(columns={"value": "US10Y"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"saved {len(frame):,} observations to {args.output}")


if __name__ == "__main__":
    main()
