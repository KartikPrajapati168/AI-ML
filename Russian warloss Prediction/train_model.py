import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

# ------------------ LOAD DATA ------------------
df = pd.read_csv('data/russia_losses_equipment.csv', parse_dates=['date'], dayfirst=True)
df.sort_values('date', inplace=True)

df.drop(columns=['greatest losses direction'], inplace=True, errors='ignore')
df.replace('', np.nan, inplace=True)
df.fillna(0, inplace=True)

equipment_cols = ['aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL',
                  'military auto', 'fuel tank', 'drone', 'naval ship',
                  'anti-aircraft warfare', 'special equipment', 'mobile SRBM system',
                  'vehicles and fuel tanks', 'cruise missiles', 'submarines',
                  'ground robotic systems']
df['total_loss'] = df[equipment_cols].sum(axis=1)
df.set_index('date', inplace=True)

print(df.isnull().sum())

# ------------------ PLOT ------------------
plt.figure(figsize=(12,6))
plt.plot(df.index, df['total_loss'], label='Daily losses')
plt.title('Daily Total Losses Over Time')
plt.xlabel('Date')
plt.ylabel('Loss count')
plt.legend()
plt.show()

# ------------------ FEATURE ENGINEERING ------------------
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofyear'] = df.index.dayofyear

for lag in [1, 2, 3, 7, 14, 30]:
    df[f'lag_{lag}'] = df['total_loss'].shift(lag)

for window in [7, 30]:
    df[f'rolling_mean_{window}'] = df['total_loss'].rolling(window).mean()
    df[f'rolling_std_{window}'] = df['total_loss'].rolling(window).std()

df.dropna(inplace=True)

# ------------------ TRAIN XGBOOST ------------------
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

X = df[['dayofweek', 'month', 'year', 'dayofyear',
        'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30']]
y = df['total_loss']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}')

# ------------------ TIME SERIES CROSS-VALIDATION ------------------
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)
    scores.append(mean_absolute_error(y_val, pred))
print(f'Cross-val MAE: {np.mean(scores):.2f} +/- {np.std(scores):.2f}')

# ------------------ PROPHET MODEL ------------------
from prophet import Prophet

prophet_df = df[['total_loss']].reset_index()
prophet_df.columns = ['ds', 'y']

model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model_prophet.fit(prophet_df)

future = model_prophet.make_future_dataframe(periods=180)
forecast = model_prophet.predict(future)

fig = model_prophet.plot(forecast)
plt.show()

future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(180)

# ------------------ SAVE MODEL ------------------
joblib.dump(model_prophet, 'prophet_model.pkl')
print("✅ Prophet model successfully saved as 'prophet_model.pkl'")