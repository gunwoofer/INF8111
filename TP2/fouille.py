#Fichier ou l'on fait les recherches pour trouver quels sont les meilleurs variable, pourquoi.
# Parciculièrement de garder une trace ici pour le rapport.
# Utiliser matplotlib et panda pour representer les informations sous forme de graph

import numpy as np
import seaborn as sns
import bixi_dataset as ds
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stats

def update_median(array):
    liste = []
    for val in array:
        if val != -1000:
            liste.append(val)
    mediane = stats.median(liste)
    for i in range(len(array)):
        if array[i] == -1000:
            array[i] = mediane
    return array

def standardize(x): 
    mean_px = X_train.mean()
    std_px = X_train.std()
    return (x-mean_px)/std_px

def reformat_data(x):
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            if x[i][j] == '':
                x[i][j] = -1000
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

X_train = reformat_data(train.iloc[:,1:6].values).astype('float32')
y_train = train.iloc[:,15].values.astype('int32')

for x in X_train:
    update_median(x)
features = standardize(X_train)
test = 5 