"""Time-aware model comparison and optional SHAP interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

FEATURES = [
    "x1_kospi200_return", "x2_kospi200_range", "x3_kospi200_abs_oc",
    "x4_futures_return", "x5_futures_range", "x6_basis",
    "x7_sp500_return", "x8_nasdaq_return", "x9_vix_change",
    "x10_usdkrw_change",
]


def evaluate(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="kospi200_model_variables.csv")
    parser.add_argument("--target", default="Y1_shock_vol")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.data, index_col=0).dropna(subset=FEATURES + [args.target])
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[FEATURES], train[args.target]
    X_test, y_test = test[FEATURES], test[args.target]

    models: dict[str, object] = {"ols": LinearRegression()}
    tscv = TimeSeriesSplit(n_splits=5)
    rf = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [300], "max_depth": [4, 8, None], "min_samples_leaf": [2, 5]},
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    models["random_forest"] = rf

    try:
        from xgboost import XGBRegressor
        xgb = GridSearchCV(
            XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=-1),
            {"n_estimators": [300, 600], "max_depth": [2, 3], "learning_rate": [0.03, 0.05]},
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        models["xgboost"] = xgb
    except ImportError:
        pass

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = evaluate(y_test, pred)
        fitted[name] = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    tree_candidates = [n for n in ("xgboost", "random_forest") if n in fitted]
    if tree_candidates:
        best_name = min(tree_candidates, key=lambda n: results[n]["rmse"])
        try:
            import shap
            explainer = shap.TreeExplainer(fitted[best_name])
            values = explainer.shap_values(X_test)
            np.save(output / f"{best_name}_shap_values.npy", np.asarray(values))
        except Exception as exc:
            print(f"SHAP export skipped: {exc}")


if __name__ == "__main__":
    main()
