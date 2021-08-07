import lr as lr
import pandas as pd
import numpy as np
import csv
import os
import scipy
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.manifold import TSNE

lm = LinearRegression()


data = pd.read_csv(r'../bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
data0 = data.fillna(data.mean())
data1 = data.dropna()
print(data.info())
print('\n')
print(pd.isna(data1))
print(pd.isna(data0))

# Preprocessing
# Open = Open price at start time window
# High = High price within time window
# Low = Low price within time window
# Close = Close price at end of time window
# Volume_(BTC) = Volume of BTC transacted in this window
# Volume_(Currency) = Volume of corresponding currency transacted in this window
# Weighted_Price = VWAP- Volume Weighted Average Price
print(data1.columns)
X = data1[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']]
y = data1[['Weighted_Price']]
print(X)
print(X.columns)

SCALER = StandardScaler()
scaler1 = SCALER.fit(data1[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']])
X_scale = SCALER.transform(data1[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']])
scaler2 = SCALER.fit(data1[['Weighted_Price']])
y_scale = SCALER.transform(data1[['Weighted_Price']])
X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale, test_size=0.2, random_state=0)

lm.fit(X_train, y_train)
test = lm.predict(X_test)
sns.residplot(x=test, y=y_test)
# plt.show()
chi2 = scipy.stats.chi2_contingency(data1[['Low', 'Close']], correction=True)
print(chi2)

# PCA
pca = PCA(n_components=4, iterated_power=1)
train_reduced = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
# plt.figure(figsize=(8, 6))
# plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_train, alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.show()

test_reduce = pca.transform(X_test)
# plt.figure(figsize=(8, 6))
# plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_test, alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.show()

TS_NE = TSNE(n_components=4, random_state=37, n_iter=6)
train_reduced = TS_NE.fit_transform(X_train)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
# plt.figure(figsize=(8, 6))
# plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_test, alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.show()

# pr = PolynomialFeatures(degree=2, include_bias=False)
# x_polly = pr.fit_transform(X_train[['Open', 'High']])

