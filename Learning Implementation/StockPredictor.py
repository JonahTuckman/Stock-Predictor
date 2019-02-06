#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 18:23:09 2019

@author: JonahTuckman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

# Normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

# Read the file
datafile = pd.read_csv('MY_NSE-TATAGLOBAL.csv')

#head
datafile.head()

## Closing price will be our target variable
# Plot the closing price vs date
datafile['Date'] = pd.to_datetime(datafile.Date, format='%Y-%m-%d')
datafile.index = datafile['Date']

# Plot it
plt.figure(figsize=(16,8))
plt.plot(datafile['Close'], label = 'Close Price History')

## Predicted closing price for each day will be the 
## average of a set or previously observed values

## FIRST STEP
# Create a dataframe that contains only the date and close price columns
# Then split it into train and validation sets 

# DataFrame with date and close
data = datafile.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(datafile)), columns = ['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    
# Split into train and validation 
# Cannot do it randomly because are in order by date
train = new_data[:987]
valid = new_data[987:]

new_data.shape
#(1234,2)
train.shape
#(987,2)
valid.shape
#(247,2)

#Make sure no overlap
train['Date'].min()
train['Date'].max()
valid['Date'].min()
valid['Date'].max()

# Time to create predictions
predictions = []
for i in range(0,247):
    a = train['Close'][len(train)-248+i:].sum() + sum(predictions)
    b = a / 248
    predictions.append(b)
    
# Root mean Square deviation
rms = np.sqrt(np.mean(np.power((np.array(valid['Close'])-predictions),2)))
rms
# 61.029.......

# Plot predicted value and actual value
valid['Predictions'] = 0
valid['Predictions'] = predictions
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
# Plot is understanding trends, not necesarily accurate though

# Using Linear Regression 

# First sort dataset, then create new one to ammend while keeping original intact
datafile['Date'] = pd.to_datetime(datafile.Date,format='%Y-%m-%d')
datafile.index = datafile['Date']

#sorting the datafile based on date
data = datafile.sort_index(ascending=True, axis=0)

#creating new dataset to be worked on 
new_data = pd.DataFrame(index = range(0,len(datafile)), columns=['Date', 'Close'])
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# create new features to be split on.
    # Year, month, dat, dayofweek, etc
from fastai.tabular import add_datepart
add_datepart(new_data, 'Date')
new_data.drop('Elapsed',axis=1, inplace= True) # Elapsed is the time stamp

# is monday or friday?
new_data['mon_fri'] = 0
for i in range(0, len(new_data)):
    if(new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i]==4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0

######################################################
## Add more features to use in predicting stock price
######################################################


# split into train and validation 
train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis = 1)
y_train = train['Close']
x_valid = valid.drop('Close', axis = 1)
y_valid = valid['Close']

# Linear Regression 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# make prediction with this linear regression 
prediction = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid)- np.array(prediction)),2)))
rms
#87.5.......

# plot to see results 
valid['Predictions'] = 0
valid['Predictions'] = prediction

valid.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])

## Poor prediction
# Over fit the date and month column, looks at data from same data a month ago,
# same date a year ago

# K nearest
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

# scaling the data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv = 5)

# fit model and make predictions 
model.fit(x_train, y_train)
prediction = model.predict(x_valid)

rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(prediction)),2)))
rms
# 105.36........

valid['Predictions'] = 0
valid['Predictions'] = predictions
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])

# Autoregressive Integrated Moving Average ARIMA
# Good for time series forecasting / predicting 
# ARIMA takes in past to predict future 
# PARAMETERS: p (past value used for future), q (past errors), d (order of differencing)

from pyramid.arima import auto_arima

data = datafile.sort_index(ascending = True, axis = 0)
train = data[:987]
valid = data[987:]

training = train['Close']
validation = valid['Close']

model = auto_arima(training, start_p = 1, start_q = 1,
                   max_p = 3, max_q = 3, m = 12, start_P = 0, seasonal = True,
                   d = 1, D = 1, trace = True, error_action = 'ignore', 
                   suppress_warnings = True)
model.fit(training)

forecast = model.predict(n_periods = 247)
forecast = pd.DataFrame(forecast, index = valid.index, columns = ['Prediction'])

rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(forecast['Prediction'])),2)))
rms

plt.plot(train['Close'])
plt.plot(valid['Close'])
plt.plot(forecast['Prediction'])


## PROPHET - input (dataFrame with date and target)
from fbprophet import Prophet

new_data = pd.DataFrame(index=range(0, len(datafile)), columns = ['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
 
new_data['Date'] = pd.to_datetime(new_data.Date, format='%Y-%m-%d')
new_data.index = new_data['Date']

new_data.rename(columns = {'Close':'y', 'Date': 'ds'}, inplace = True)

train = new_data[:987]
valid = new_data[987:]

model = Prophet()
model.fit(train)

close_prices = model.make_future_dataframe(periods=len(valid))
forecast = model.predict(close_prices)

forecast_valid = forecast['yhat'][987:]
rms = np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))
rms

# 93.9.......

valid['Predictions'] = 0
valid['Predictions'] = forecast_valid.values

plt.plot(train['y'])
plt.plot(valid[['y', 'Predictions']])

# Still following upwards trend as drop is unseen


# Long Short Term Memory LSTM
# input gate: info to cell state
# forget gate: removes no longer important info
# output gate: selects output

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = datafile.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(datafile)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms = np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms


train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])



