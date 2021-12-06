# -*- coding: utf-8 -*-
# IMPORT LIBRARIES AND PACKAGES
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt

# %%
# IMPORT DATASETS
traindata = pd.read_csv('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/THESIS WRITING/programs/corrindiv31train.csv')
# print('TRAINING DATASET \n:', traindata.describe().transpose())
X_train = traindata.drop('Y', axis=1)
y_train = traindata['Y']
#print(sns.countplot(x = 'Y', data=traindata))

testdata = pd.read_csv('C:/Users/onjad/OneDrive/Documents/RESEARCH PHD/THESIS WRITING/programs/corrindiv31test.csv')
# print('TEST DATASET \n:', testdata.describe().transpose())
# print('TEST DATASET \n:', testdata)
X_test = testdata.drop('Y', axis=1)
y_test = testdata['Y']

print('Length of X (train): {} | Length of Y (train): {}'.format(len(X_train), len(y_train)))
print('Length of X (test): {} | Length of Y (test): {}'.format(len(X_test), len(y_test)))

# %%
# DATA PREPROCESSING
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#print('\n\n XTRAIN: \n', X_train)
#print('\n\n XTEST: \n', X_test)

# %%
# IMPLEMENT NEURAL NETWORK

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
print(X_train.shape)

# BINARY CLASSIFICATION
n_inputs = X_train.shape[1]

model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(31, activation='relu'),
    Dense(1, activation='sigmoid')
])

# %%
# COMPILE THE NN
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# %%
# FIT THE NN TO THE TRAINING SET
model.fit(X_train, y_train, batch_size=40, validation_data=(X_test, y_test),
          epochs=1000, verbose=2, callbacks=[early_stop])

# %%
# PLOT THE LOSS
losses = pd.DataFrame(model.history.history)
losses.plot()

# %%
# PREDICTION (O if bankrupt and 1 if healthy)
import numpy as np
y_pred = model.predict(X_test)

pred = []

for predictions in y_pred:
    if predictions < 0.50:
        predictions = 0
    else:
        predictions=1
    pred.append(predictions)
    
y_pred = np.array(pred)
y_test = np.array(y_test)

print(y_pred)
print(y_test)

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('\n\nAccuracy score is:', accuracy_score(y_test, y_pred), '\n')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.show()

# %%
# SAVE WEIGHTS
weights = model.get_weights()
# for layer in model.layers:
#     weights = layer.get_weights()

import sys
sys.stdout = open('indiv_weights.txt', 'w')
print("\n\nweights:\n", weights)


