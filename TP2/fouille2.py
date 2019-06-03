#Fichier ou l'on fait les recherches pour trouver quels sont les meilleurs variable, pourquoi.
# Parciculièrement de garder une trace ici pour le rapport.
# Utiliser matplotlib et panda pour representer les informations sous forme de graph

import numpy as np
import seaborn as sns
import bixi_dataset as ds
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stats
import tensorflow as tf
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras import backend as K
from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier

def construct_layers():
    # number of layers
    classifier = keras.Sequential()
    classifier.add(keras.layers.Dense(units=5, activation="relu", input_dim=188))
    classifier.add(keras.layers.Dense(units=5, activation="relu"))

    classifier.add(keras.layers.Dense(units=1, activation="sigmoid"))
    classifier.compile(
    optimizer='adamax',
    loss='mean_squared_error',
    metrics=["accuracy"])
    return classifier


def station_to_one_hot_vector(station_code):
    dic_station_code = dict.fromkeys(station_code.tolist(), 0)
    i = 1
    for key, value in dic_station_code.items():
        dic_station_code[key] = i
        i += 1
    list_one_hot = []
    for k in range (len(station_code)):
        y = dic_station_code[station_code[k]]
        one_hot = [0] * i
        one_hot[y-1] = 1
        list_one_hot.append(one_hot)
    return np.array(list_one_hot)

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


train = pd.read_csv("data/training.csv")
print(train.shape)
train.head()

test= pd.read_csv("data/test.csv")
print(test.shape)
test.head()

_, _, id_test = ds.getData(ds.TEST_FILE_NAME, type_data='test')

print('formater donnée')
X_train = reformat_data(train.iloc[:,1:6].values).astype('float32')
X_train = update_median(X_train)

test_val = update_median(reformat_data(test.iloc[:,1:6].values).astype('float32'))
station_code_test = test.iloc[:, 13].values

station_code = train.iloc[:, 13].values


####
print('creer station code')
station_code_one_hot = station_to_one_hot_vector(station_code)
station_code_one_hot_test = station_to_one_hot_vector(station_code_test)

volume = train.iloc[:, 15].values

print('concatene')
train_x = np.concatenate((X_train, station_code_one_hot), axis=1)
test_x = np.concatenate((test_val, station_code_one_hot_test), axis=1)

print('arbre de decision')
clf = DecisionTreeClassifier(random_state=0)
clf.fit(train_x, volume)

print('debut prediction')
valid_pred = clf.predict(train_x)
f_score = f1_score(volume, valid_pred)
print('fscore : ' + str(f_score))

classifier = KerasClassifier(build_fn=construct_layers, batch_size=20, epochs=1)
classifier.fit(x=train_x, y=volume)
pred = classifier.predict(train_x)
f_score = f1_score(volume, pred)

print('fscore neural network : ' + str(f_score))
test_pred = clf.predict(test_x)
ds.writeCsv(id_test, test_pred)

# plt.hist(volume, range = (0, 1), bins = 2, color = 'blue',
#             edgecolor = 'red')
# plt.xlabel('volume')
# plt.ylabel('nb volume')
# plt.title('Repartission des volumes')
# plt.show()

# a = []
# for station in station_code:
#     a.append(dic_station_code[station])
# df = pd.DataFrame({'A': volume, 'B': a})

# df[df['A']==1].hist('B', bins=189)
# plt.xlabel('catégorie volume')
# plt.ylabel('nb volume à 1')
# plt.title('Repartition des volumes ayant une valeur de 1')
# df[df['A']==0].hist('B', bins=189)
# plt.xlabel('catégorie volume')
# plt.ylabel('nb volume à 0')
# plt.title('Repartission des volumes ayant une valeur de 0')

# plt.show()