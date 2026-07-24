import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
import copy

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data/processed/Vol_Pred_Final_KST_Data_v2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df[df['CLSPRC_IDX'] < 700].reset_index(drop=True)

df['Intraday_Abs_Ret'] = np.abs(np.log(df['CLSPRC_IDX'] / df['OPNPRC_IDX']))
df['US10Y_chg'] = df['US10Y'] - df['US10Y'].shift(1)
for lag in [1, 2, 3]:
    df[f'RV_Proxy_lag{lag}'] = df['RV_Proxy'].shift(lag)
    df[f'log_ret_kospi_lag{lag}'] = df['log_ret_kospi'].shift(lag)
    df[f'Trade_Vol_lag{lag}'] = df['ACC_TRDVOL'].shift(lag)
    df[f'Intraday_Abs_Ret_lag{lag}'] = df['Intraday_Abs_Ret'].shift(lag)
for lag in [1, 2]:
    df[f'sp500_ret_lag{lag}'] = df['sp500_ret'].shift(lag)
    df[f'nasdaq_ret_lag{lag}'] = df['nasdaq_ret'].shift(lag)
    df[f'vix_chg_lag{lag}'] = df['vix_chg'].shift(lag)
    df[f'fx_ret_lag{lag}'] = df['fx_ret'].shift(lag)
df['RV_Proxy_roll_mean_5'] = df['RV_Proxy'].shift(1).rolling(window=5).mean()
df['RV_Proxy_roll_std_5'] = df['RV_Proxy'].shift(1).rolling(window=5).std()
df['log_ret_kospi_roll_mean_5'] = df['log_ret_kospi'].shift(1).rolling(window=5).mean()

feature_cols = [
    'sp500_ret', 'nasdaq_ret', 'vix_chg', 'fx_ret', 'US10Y_chg',
    'sp500_ret_lag1', 'sp500_ret_lag2', 'nasdaq_ret_lag1', 'nasdaq_ret_lag2',
    'vix_chg_lag1', 'vix_chg_lag2', 'fx_ret_lag1', 'fx_ret_lag2',
    'RV_Proxy_lag1', 'RV_Proxy_lag2', 'RV_Proxy_lag3',
    'log_ret_kospi_lag1', 'log_ret_kospi_lag2', 'log_ret_kospi_lag3',
    'Trade_Vol_lag1', 'Trade_Vol_lag2', 'Trade_Vol_lag3',
    'Intraday_Abs_Ret_lag1', 'Intraday_Abs_Ret_lag2', 'Intraday_Abs_Ret_lag3',
    'RV_Proxy_roll_mean_5', 'RV_Proxy_roll_std_5', 'log_ret_kospi_roll_mean_5'
]
target_col = 'log_target_y2'
df_model = df[['Date'] + feature_cols + [target_col]].dropna().reset_index(drop=True)
X = df_model[feature_cols]
y = df_model[target_col]
split_idx = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df_model['Date'].iloc[split_idx:].reset_index(drop=True)

lr_model = LinearRegression().fit(X_train, y_train)
rf_grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {'n_estimators':[100,200,300], 'max_depth':[3,4,5], 'max_features':['sqrt','log2',0.5]},
    cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error', n_jobs=1
)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_

xgb_grid_search = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    {'n_estimators':[100,200,300], 'learning_rate':[0.01,0.05], 'max_depth':[3,4],
     'colsample_bytree':[0.7,1.0], 'subsample':[0.7,1.0], 'reg_lambda':[1,5,10]},
    cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error', n_jobs=-1
)
xgb_grid_search.fit(X_train, y_train)
best_xgb_model = xgb_grid_search.best_estimator_

models = {'OLS':lr_model, 'Random Forest':best_rf_model, 'XGBoost':best_xgb_model}
for name, model in models.items():
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    print(name)
    print('Train MAE/RMSE:', mean_absolute_error(y_train,pred_train), np.sqrt(mean_squared_error(y_train,pred_train)))
    print('Test MAE/RMSE:', mean_absolute_error(y_test,pred_test), np.sqrt(mean_squared_error(y_test,pred_test)))

for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(X), 1):
    model = clone(best_xgb_model)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[val_idx])
    print(f'Fold {fold}: MAE={mean_absolute_error(y.iloc[val_idx], pred):.5f}, RMSE={np.sqrt(mean_squared_error(y.iloc[val_idx], pred)):.5f}')

explainer = shap.Explainer(best_xgb_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.tight_layout(); plt.show()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout(); plt.show()

extreme_idx = y_test.argmax()
local_exp = copy.deepcopy(shap_values[extreme_idx])
local_exp.values *= 1000
local_exp.base_values *= 1000
print('Local analysis date:', dates_test.iloc[extreme_idx].strftime('%Y-%m-%d'))
shap.plots.waterfall(local_exp, show=False)
plt.tight_layout(); plt.show()
