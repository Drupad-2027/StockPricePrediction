import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st

# Load the trained model
model = load_model(r'C:\Users\Drupad Dhamdhere\SPP.h5')

# Streamlit header
st.header('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input('Enter stock symbol', 'TALK')

# Date input for start and end dates
start = st.date_input('Start date', pd.to_datetime('2012-01-01'))
end = st.date_input('End date', pd.to_datetime('2022-12-31'))

# Download stock data
data = yf.download(stock, start=start, end=end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Prepare training and testing data
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Prepare data for prediction
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plotting Moving Averages
st.subheader('Price vs MA50')
MA_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(MA_50_days, 'r', label='MA 50 Days')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
MA_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(MA_50_days, 'r', label='MA 50 Days')
plt.plot(MA_100_days, 'b', label='MA 100 Days')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
MA_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(MA_100_days, 'r', label='MA 100 Days')
plt.plot(MA_200_days, 'b', label='MA 200 Days')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Make predictions
predict = model.predict(x)
scale = 1 / scaler.scale_

predict = predict * scale
y = y * scale

# Plot original vs predicted prices
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'b', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("Drupad Dhamdhere")
st.markdown("37014")
st.markdown("This application uses historical stock data to predict future prices using a trained model.")