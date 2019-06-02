# import torch
# from torch.autograd import Variable     
# import torch.nn as nn 
import bixi_dataset as ds
# import bixi_network as net
# import utils_train as utils
import tensorflow as tf
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras import backend as K
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# K.tensorflow_backend._get_available_gpus()

# config = tf.ConfigProto( device_count = {'GPU': 6 , 'CPU': 1} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def construct_layers():
    # number of layers
    classifier = keras.Sequential()
    classifier.add(keras.layers.Dense(units=5, activation="relu", input_dim=2))
    classifier.add(keras.layers.Dense(units=5, activation="relu"))

    classifier.add(keras.layers.Dense(units=1, activation="sigmoid"))
    classifier.compile(
    optimizer='adamax',
    loss='mean_squared_error',
    metrics=["accuracy"])
    return classifier

def accuracy(y, y_pred):
    sum_accuracy = 0
    for i in range(0, len(y_pred)):
        if (y[i] == y_pred[i]):
            sum_accuracy += 1
    return sum_accuracy / len(y_pred)

def main():
    print("TP2..")
    # cuda = torch.device('cuda') 
    #### CREATION DES MATRICE DE FEATURES/TARGET EN TRAIN, VALIDATION ET TEST


    train_X, train_Y, id_train = ds.getData(ds.TRAIN_FILE_NAME)
    index = int(0.6 * train_X.shape[0])
    validation_X, validation_Y = train_X[index:], train_Y[index:]
    train_X, train_Y = train_X[:index], train_Y[:index]

    test, _, id_test = ds.getData(ds.TEST_FILE_NAME, type_data='test')
    
    # classifier = KerasClassifier(build_fn=construct_layers, batch_size=20, epochs=1)
    # classifier.fit(x=train_X, y=train_Y)
    # pred = classifier.predict(validation_X[:])
    # precision = accuracy(validation_Y, pred)


    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_X, train_Y)

    valid_pred = clf.predict(validation_X)
    f_score = f1_score(validation_Y, valid_pred)
    print('fscore : ' + str(f_score))

    pred_test = clf.predict(test)

    # pred_test = classifier.predict(test)
    # final_pred = []
    # for pred_val in pred_test:
    #     if pred_val[0] == 1:
    #         final_pred.append(0)
    #     else:
    #         final_pred.append(1)
    ds.writeCsv(id_test, pred_test)


if __name__ == "__main__":
    main()



# UTiliser les arbres de décisions ? https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html