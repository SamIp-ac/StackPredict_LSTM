import lr as lr
import pandas as pd
import numpy as np
import csv
import math
# import os
# import scipy
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
import seaborn as sns
# from sklearn.manifold import TSNE
from tensorflow import keras
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense

df = pd.read_csv(r'../Stock_MRIN.csv', skip_blank_lines=True)

data = df.filter(['Close'])
data = data.dropna()
dataset = data.values
training_data_len = math.ceil(len(dataset)*.8)
scaled = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaled.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]
X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(X_train)
        print(y_train)
        print()

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)
# period = np.array(range(1, 2082)).reshape(-1, 1)

# X_train, X_test, y_train, y_test = train_test_split(period, scaled_data, test_size=0.3, random_state=0)
# X_train, y_train = np.array(X_train), np.array(y_test)
# X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)

test_data = scaled_data[training_data_len - 60:, :]
X_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60:i, 0])

X_test = np.array(X_test)
print(X_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
prediction = model.predict(X_test)
prediction = scaled.inverse_transform(prediction)

rmse = metrics.mean_squared_error(prediction, y_test, squared=False)
print(rmse)

train = data[:training_data_len]
vaild = data[training_data_len:]
vaild['Predictions'] = prediction

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Data')
plt.ylabel('Closing Price (USD)')
plt.plot(train['Close'])
plt.plot(vaild[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'])
plt.show()
print(vaild)
plt.close()
####
sns.residplot(x='Close', y='Predictions', data=vaild)
plt.show()
