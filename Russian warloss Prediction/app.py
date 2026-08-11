import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import joblib

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/russia_losses_equipment.csv', parse_dates=['date'], dayfirst=True)
    
    # ✅ SABSE PEHLE: Text wale column ko hatao (taaki fillna error na aaye)
    df.drop(columns=['greatest losses direction'], inplace=True, errors='ignore')
    
    # Empty strings ko NaN mein badlo
    df.replace('', np.nan, inplace=True)
    
    # Ab saare columns numeric hain, toh 0 se fill karo safely
    df.fillna(0, inplace=True)
    
    # Equipment columns (jinhe sum karna hai)
    equipment_cols = ['aircraft', 'helicopter', 'tank', 'APC', 'field artillery', 'MRL',
                      'military auto', 'fuel tank', 'drone', 'naval ship',
                      'anti-aircraft warfare', 'special equipment', 'mobile SRBM system',
                      'vehicles and fuel tanks', 'cruise missiles', 'submarines',
                      'ground robotic systems']
    
    # Total loss calculate karo
    df['total_loss'] = df[equipment_cols].sum(axis=1)
    
    # Prophet ke liye sirf date aur y column rakho
    df = df[['date', 'total_loss']]
    df.columns = ['ds', 'y']
    return df

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_prophet_model():
    return joblib.load('prophet_model.pkl')

# ------------------ APP UI ------------------
st.title('Russian Equipment Loss Forecast')
st.write('Predict future losses based on historical daily data')

horizon = st.sidebar.slider('Forecast Horizon (days)', min_value=30, max_value=365, value=180, step=30)

df = load_data()
model = load_prophet_model()

# Future forecast generate karo
future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

# Plot karo
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical'))
future_forecast = forecast[forecast['ds'] > df['ds'].max()]
fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'],
                         mode='lines', name='Forecast', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'],
                         fill=None, mode='lines', line_color='rgba(0,0,0,0)', name='Upper'))
fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'],
                         fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                         name='Confidence Interval (80%)'))

fig.update_layout(title='Total Loss Forecast', xaxis_title='Date', yaxis_title='Loss Count')
st.plotly_chart(fig, use_container_width=True)

# Table dikhao
st.subheader('Predicted Values (next 30 days)')
st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(30))

# Download CSV
csv = future_forecast.to_csv(index=False)
st.download_button('Download Forecast CSV', data=csv, file_name='forecast.csv', mime='text/csv')