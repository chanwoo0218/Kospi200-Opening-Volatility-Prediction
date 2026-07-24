# Presentation Summary

## Research question

Can information available before 07:59 KST predict the KOSPI 200 opening shock and the first 30 minutes of realized volatility?

## Targets

- `Y1_shock_vol`: absolute opening gap relative to the previous close
- `log_target_y2`: log-transformed realized volatility from 09:00 to 09:30

## Experimental design

- U.S. market close is shifted by one calendar day to align with the next Korean opening.
- Domestic variables are lagged by 1–3 days.
- Rolling statistics use past observations only.
- The last 20% of observations form the holdout set.
- TimeSeriesSplit is used for expanding-window cross-validation.

## Main results

| Target | Best model | MAE | RMSE |
|---|---|---:|---:|
| Y1 | XGBoost | 0.01714 | 0.02382 |
| Y2 | XGBoost | 0.00481 | 0.00619 |

The Y2 model reduced MAE by about 21% relative to the naive baseline.

## Interpretation

- Y1 was driven mainly by recent KOSPI return, volume, and short rolling averages.
- Y2 was driven mainly by lagged realized volatility and its five-day average.
- Extreme dates were explained with local SHAP waterfall plots.

## Public-release note

The submitted presentation PDF is preserved in the downloadable reviewed package supplied with this repository update. The GitHub connector used in this session supports text commits but did not expose a safe direct handoff for the uploaded binary PDF, so this Markdown summary is committed instead.
