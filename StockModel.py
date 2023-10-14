import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as ds
import datetime as dt
import streamlit as st


from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

start=dt.datetime(2011,1,1)
end=dt.datetime(2020,1,1)

st.title('Stock Market Prediction')

input=st.text_input("Enter Stock Ticker Symbol (AAPL)")
data=ds.DataReader(input,'yahoo',start,end)


#Normal CLosing Price
st.subheader('Normal Closing Price')
plt.plot(data.Close)
plt.title(f"{input} Share Closing Price")
plt.xlabel('Year')
plt.ylabel('Share Price')
st.pyplot(plt)

#Prepare Data
scal=MinMaxScaler(feature_range=(0,1))
scalData=scal.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days=60


#Loading the Model

model= keras.models.load_model('stock_model.h5')

#Load test data
test_start=dt.datetime(2018,1,1)
test_end=dt.datetime.now()

test_data=ds.DataReader(input, 'yahoo', test_start, test_end)
actual_price= test_data['Close'].values

total_dataset= pd.concat((data['Close'], test_data['Close']), axis=0)

model_input= total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_input=model_input.reshape(-1,1)
model_input= scal.transform(model_input)

#make prediction on test data

x_test= []

for x in range(prediction_days, len(model_input)):
    x_test.append(model_input[x-prediction_days:x, 0])

x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_price=model.predict(x_test)
predicted_price=scal.inverse_transform(predicted_price)

#Plot the Test prediction
st.subheader('Actual VS Predicted Price(2018-Current)')
fig2=plt.figure(figsize=(12,8))
plt.plot(actual_price, 'b', label= 'Original Price')
plt.plot(predicted_price,'r', label='Predicted Price')
plt.title(f"{input} Share Price")
plt.xlabel('Days')
plt.ylabel('Share Price')
plt.legend(loc='best')
st.pyplot(fig2)

#Predict Next

df=test_data.filter(['Close'])

last_60=df[-60:].values
scaled_last=scal.transform(last_60)

x_test= []
x_test.append(scaled_last)
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predict_price=model.predict(x_test)
predict_price=scal.inverse_transform(predict_price)

st.write(f"Tomorrow closing price is:{predict_price}")
