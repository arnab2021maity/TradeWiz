# import numpy as np
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model
# import streamlit as st
# import matplotlib.pyplot as plt

# model = load_model(r'C:\\Users\\DELL\\ML project\\Stock market predictor\\Stock Predictions Model.keras')


# st.header('Stock Market Predictor')

# stock =st.text_input('Enter Stock Symnbol', 'GOOG')
# start = '2012-01-01'
# end = '2022-12-31'

# data = yf.download(stock, start ,end)

# st.subheader('Stock Data')
# st.write(data)

# data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
# data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

# pas_100_days = data_train.tail(100)
# data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
# data_test_scale = scaler.fit_transform(data_test)

# st.subheader('Price vs MA50')
# ma_50_days = data.Close.rolling(50).mean()
# fig1 = plt.figure(figsize=(8,6))
# plt.plot(ma_50_days, 'r')
# plt.plot(data.Close, 'g')
# plt.show()
# st.pyplot(fig1)

# st.subheader('Price vs MA50 vs MA100')
# ma_100_days = data.Close.rolling(100).mean()
# fig2 = plt.figure(figsize=(8,6))
# plt.plot(ma_50_days, 'r')
# plt.plot(ma_100_days, 'b')
# plt.plot(data.Close, 'g')
# plt.show()
# st.pyplot(fig2)

# st.subheader('Price vs MA100 vs MA200')
# ma_200_days = data.Close.rolling(200).mean()
# fig3 = plt.figure(figsize=(8,6))
# plt.plot(ma_100_days, 'r')
# plt.plot(ma_200_days, 'b')
# plt.plot(data.Close, 'g')
# plt.show()
# st.pyplot(fig3)

# x = []
# y = []

# for i in range(100, data_test_scale.shape[0]):
#     x.append(data_test_scale[i-100:i])
#     y.append(data_test_scale[i,0])

# x,y = np.array(x), np.array(y)

# predict = model.predict(x)

# scale = 1/scaler.scale_

# predict = predict * scale
# y = y * scale

# st.subheader('Original Price vs Predicted Price')
# fig4 = plt.figure(figsize=(8,6))
# plt.plot(predict, 'r', label='Original Price')
# plt.plot(y, 'g', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.show()
# st.pyplot(fig4)


import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model(r'C:\\Users\\DELL\\ML project\\Stock market predictor\\Stock Predictions Model.keras')

# Streamlit app header
st.header('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Split data into training and testing sets
data_train = data['Close'][:int(len(data) * 0.80)]
data_test = data['Close'][int(len(data) * 0.80):]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(np.array(data_train).reshape(-1, 1))

# Prepare the test data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(np.array(data_test).reshape(-1, 1))

# Prepare the training and testing sets
x_test = []
y_test = []

for i in range(100, len(data_test_scaled)):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict the stock prices
predictions = model.predict(x_test)

# Inverse transform the predictions and actual values
scale = 1 / scaler.scale_[0]
predictions = predictions * scale
y_test = y_test * scale

# Plot the results
st.subheader('Original Price vs Predicted Price')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test, 'g', label='Original Price')
ax.plot(predictions, 'r', label='Predicted Price')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()

st.pyplot(fig)

# Moving Averages Plots
st.subheader('Price vs MA50')
ma_50_days = data['Close'].rolling(window=50).mean()
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(data['Close'], 'g', label='Close Price')
ax1.plot(ma_50_days, 'r', label='MA50')
ax1.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data['Close'].rolling(window=100).mean()
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(data['Close'], 'g', label='Close Price')
ax2.plot(ma_50_days, 'r', label='MA50')
ax2.plot(ma_100_days, 'b', label='MA100')
ax2.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data['Close'].rolling(window=200).mean()
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(data['Close'], 'g', label='Close Price')
ax3.plot(ma_100_days, 'r', label='MA100')
ax3.plot(ma_200_days, 'b', label='MA200')
ax3.legend()
st.pyplot(fig3)
