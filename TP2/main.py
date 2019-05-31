# import torch
# from torch.autograd import Variable     
# import torch.nn as nn 
import bixi_dataset as ds
# import bixi_network as net
import utils_train as utils
import tensorflow as tf
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

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


    # train_X, train_Y, id_train = ds.getData(ds.TRAIN_FILE_NAME)
    # index = int(0.6 * train_X.shape[0])
    # validation_X, validation_Y = train_X[index:], train_Y[index:]
    # train_X, train_Y = train_X[:index], train_Y[:index]

    test, _, id_test = ds.getData(ds.TEST_FILE_NAME, type_data='test')

    # batch_size = 100
    # test_batch_size = 100

    # train_X = torch.from_numpy(train_X)#changer le device si vous utilisez pas cuda
    # train_Y = torch.from_numpy(train_Y)# changer le device si vous utilisez pas cuda
    # dataset = torch.utils.data.TensorDataset(train_X, train_Y)

    # train_loader = torch.utils.data.DataLoader(dataset,
    #     batch_size=batch_size, shuffle=True)

    # valid_loader = torch.utils.data.DataLoader((validation_X, validation_Y),
    #     batch_size=batch_size, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(test,
    #     batch_size=batch_size, shuffle=True)

    print('chargement des données fini')
    
    
    
    ########## PYTORCH #################




    # print('debut des experimentations')
    # best_precision = 0
    # for model in [net.BixiNetwork()]:  # Liste des modeles que l'on veut utiliser pour tester. Mettez à jour pour éviter de perdre du temps
    #     #model.cuda()  # Commenter cette ligne si vous avez pas cuda
    #     model.double()
    #     model, precision = utils.experiment(model, train_loader, valid_loader)
    #     if precision > best_precision:
    #         best_precision = precision
    #         best_model = model
    #         print('nouveau model detecte meilleur avec une precision de : ' + best_precision)


    ########## TF, parce que fuck les bug pythorch ###########

    # classifier = KerasClassifier(build_fn=construct_layers, batch_size=20, epochs=1)
    # classifier.fit(x=train_X, y=train_Y)
    # pred = classifier.predict(validation_X[:])
    # precision = accuracy(validation_Y, pred)

    # pred_test = classifier.predict(test[:])
    pred_test = []
    for i in range (len(id_test)):
        pred_test.append(0)
    ds.writeCsv(id_test, pred_test)

def construct_layers():
    # number of layers
    classifier = keras.Sequential()
    classifier.add(keras.layers.Dense(units=9, activation="relu", input_dim=9))
    classifier.add(keras.layers.Dense(units=9, activation="relu"))

    classifier.add(keras.layers.Dense(units=1, activation="softmax"))
    classifier.compile(
    optimizer='adamax',
    loss='mean_squared_error',
    metrics=["accuracy"])
    return classifier
if __name__ == "__main__":
    main()



# UTiliser les arbres de décisions ? https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html