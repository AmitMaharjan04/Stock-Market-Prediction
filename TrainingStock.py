import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as ds
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


#load company
org = 'FB'

start=dt.datetime(2011,1,1)
end=dt.datetime(2018,1,1)

data=ds.DataReader(org,'yahoo',start,end)

#Prepare Data
scal=MinMaxScaler(feature_range=(0,1))
scalData=scal.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days=60

x_train = []
y_train = []

for x in range(prediction_days, len(scalData)):
    x_train.append(scalData[x-prediction_days:x,0])
    y_train.append(scalData[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train=np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


#Build The Model
model= Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Prediction of next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

model.save('stock_model.h5')
