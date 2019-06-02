#Fichier ou l'on fait les recherches pour trouver quels sont les meilleurs variable, pourquoi.
# Parciculièrement de garder une trace ici pour le rapport.
# Utiliser matplotlib et panda pour representer les informations sous forme de graph

import numpy as np
import seaborn as sns
import bixi_dataset as ds
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stats

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def update_median(array):
    col_median = np.nanmean(array, axis=0)
    inds = np.where(np.isnan(array))
    array[inds] = np.take(col_median, inds[1])
    return array

def standardize(x): 
    mean_px = X_train.mean()
    std_px = X_train.std()
    return (x-mean_px)/std_px

def reformat_data(x):
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            if x[i][j] == '':
                x[i][j] = np.nan
            else:
                x[i][j] = float(str(x[i][j]).replace(',', '.'))
    return x

columns = ['Temperature (°C)', 'Drew point (°C)', 'volume']

train = pd.read_csv("data/training.csv")
print(train.shape)
train.head()

test= pd.read_csv("data/test.csv")
print(test.shape)
test.head()
np.nan
X_train = reformat_data(train.iloc[:,1:6].values).astype('float32')
y_train = train.iloc[:,15].values.astype('int32')
X_train = update_median(X_train)
# y_train = y_train[~np.isnan(X_train).any(axis=1)]
# X_train = X_train[~np.isnan(X_train).any(axis=1)]
y_train = y_train.reshape((len(y_train), 1))
features = standardize(X_train)

train_val = np.concatenate((features, y_train), axis=1)

# for x in X_train:
#     update_median(x)
test = 5 

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

valid_pred = clf.predict(X_train)
f_score = f1_score(y_train, valid_pred)
print('fscore : ' + str(f_score))

pred_test = clf.predict(test)