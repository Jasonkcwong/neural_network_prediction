#!/usr/bin/env python
# coding: utf-8

# In[]:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from keras.optimizers import SGD
from keras import metrics


# read the csv file with the heather to have an idea about the dataset
df = pd.read_csv('Franchise_dataset.csv')
df.head()

biz_types = df[['Business Type']]
# use one hot encoding to reshape business type
ohe = OneHotEncoder()
ohe.fit(biz_types)
transformed = ohe.transform(biz_types)

dataset = np.concatenate((
	df[['Net Profit','Counter Sales','Drive-through Sales']], 
	transformed.toarray()), axis=1)

print(dataset)

# In[]:

X = dataset[:,1:6]
y = dataset[:,0:1]


# In[]:

# define the keras sequential model
model = Sequential()
# add one hidden layers with 20 nodes and 5 input variables
model.add(Dense(20, input_shape=(5,), activation='relu'))

# define the output layer with one node that uses ELU activation function
model.add(Dense(1, activation='elu'))


# In[]:

# compile the keras model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[metrics.RootMeanSquaredError()])
model.summary()

# In[]:


# fit the keras model on the dataset
#hist = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))
model.fit(X, y, epochs=200, batch_size=10)
# if you get error running this in Jupyter user verbose=0 when you call model.fit function


# In[]:

_, rmse = model.evaluate(X, y)
print('Root Mean Squared Error: %4f' % rmse)

# make regression predictions with the model
predictions = model.predict(X)
# summarize the test data prediction
for i in range(5):
	print(f'Test Data {X[i].tolist()} => {predictions[i]} (expected {y[i]})')

# predict data => 0.5m counter sales, 0.7m drive-through sales, pizza store
predict_data = [[0.5, 0.7, 0, 0, 1]]
prediction = model.predict(predict_data)

print(f'Input of prediction => {predict_data} Output of prediction => {prediction}')
