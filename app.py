import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from datetime import date
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

stocks = ('GOOGL', 'AAPL', 'MSFT', 'GME')
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title('Stock Closing Price Prediction')
user_input = st.text_input('Enter Stock Ticker', 'GOOGL')
user_input = st.selectbox('Select dataset for prediction', stocks)
df = yf.download(user_input, START, TODAY)
st.subheader('Dated from 1st Jan, 2009 to 1st Jan, 2023')
st.write(df.describe())
st.subheader('Closing Price Vs Time Chart')
fig1 = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(fig1)
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
st.subheader('Closing Price Vs Time Chart with 100 days Moving Average')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(df.Close, 'r', label="Per Day Closing")
plt.plot(ma100, 'g', label="Moving Average 100")
st.pyplot(fig2)
st.subheader('Closing Price Vs Time Chart with 100 days and 200 days Moving Average')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(ma200, 'b', label="Moving Average 200")
plt.plot(ma100, 'g', label="Moving Average 100")
st.pyplot(fig2)
train_df = pd.DataFrame(df['Close'][0: int(len(df)*0.85)])
test_df = pd.DataFrame(df['Close'][int(len(df)*0.85):int(len(df))])
scaler = MinMaxScaler(feature_range=(0, 1))
train_df_arr = scaler.fit_transform(train_df)
model = load_model('keras_model.h5')
past_100_days = train_df.tail(100)
final_df = past_100_days._append(test_df, ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])    
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scale = scaler.scale_
scale_factor = 1/scale[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor
st.subheader('Predicted Vs Original')
fig3 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'g', label="Original Price")
plt.plot(y_pred, 'r', label="Predicted Price")
st.pyplot(fig3)