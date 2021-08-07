import seaborn as sns
import pandas as pd
import numpy as np
import math
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense

df = pd.read_csv(r'../0823.HK.csv', skip_blank_lines=True)
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
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

prediction = model.predict(X_test)
prediction = scaled.inverse_transform(prediction)

rmse = metrics.mean_squared_error(prediction, y_test, squared=False)

train = data[:training_data_len]
vaild = data[training_data_len:]
vaild['Predictions'] = prediction

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Data')
plt.ylabel('Closing Price (USD)')
plt.plot(train['Close'])
plt.plot(vaild[['Close', 'Predictions']])
plt.legend(['Train', 'Real', 'Predictions'])
plt.show()
plt.close()

print(vaild)

######
link = pd.read_csv(r'../0823.HK.csv', skip_blank_lines=True)

new_df = link.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaled.transform(last_60_days)

x_test = []
x_test.append(last_60_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred_price = model.predict(x_test)
pred_price = scaled.inverse_transform(pred_price)

R2 = metrics.r2_score(prediction, y_test)
print('The R-squared score is : ', R2)
print('The root mean squared error is : ', rmse)
print('The closing on the next day is : ', pred_price)

sns.residplot(x=prediction, y=y_test, data=vaild)
plt.show()
