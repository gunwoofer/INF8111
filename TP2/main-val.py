import bixi_dataset as ds
import torch.nn.functional as F
import random
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from torch import mean, std, from_numpy, save, load
from preprocess import get_station_code, categorizeTemperatures, makeHotVector, categorizeDrewPoint, get_month
from bixi_network import BixiNetwork
from sklearn.metrics import f1_score, fbeta_score, accuracy_score, classification_report, confusion_matrix  
from sklearn.tree import DecisionTreeClassifier


LEARNING_RATE = 0.1
NB_EPOCH = 10

def main():
    # On recupere les donnees d'entrainement
    print("Recuperation des donnees d'entrainement..")
    train_X, train_Y, id_train = ds.getData(ds.TRAIN_FILE_NAME)
    print('nb de 1 : ' + str(train_Y.sum()))
    # On recupere les donnees de test a predire
    print("Recuperation des donnees de test..")
    test_X, _, id_test = ds.getData(ds.TEST_FILE_NAME, type_data='test')

    # Preprocessing
    train_X = preprocessPipeline(train_X)
    test_X = preprocessPipeline(test_X)

    # On divise en set de validation et d'entrainement
    print("Division du set d'entrainement en validation-entrainement..")
    index = int(0.6 * train_X.shape[0])

    shuf = list(zip(train_X, train_Y))
    random.shuffle(shuf)

    train_X, train_Y = zip(*shuf)

    validation_X, validation_Y = train_X[index:], train_Y[index:]
    train_X, train_Y = train_X[:index], train_Y[:index]
    validation_X, validation_Y = np.array(validation_X), np.array(validation_Y)
    # On normalise les données
    # print("Normalisation des données..")
    # train_X = normalize(train_X)
    # validation_X = normalize(validation_X)
    # test_X = normalize(test_X)

    print('arbre de decision')
    print('entrainement sur ' + str(len(train_X)) + ' données')
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf.fit(train_X, train_Y)

    print('debut prediction')
    valid_pred = clf.predict(validation_X)
    valid_pred = proba2result(validation_Y).astype('uint8')
    f_score = f1_score(validation_Y, valid_pred)
    accuracy_arbre = accuracy_score(validation_Y, valid_pred)
    print('fscore arbre de decision (validation) : ' + str(f_score))
    print('accuracy arbre de decision (validation) : ' + str(accuracy_arbre))
    print("on trouve " + str(valid_pred.sum()) + " valeur ayant une valeur de 1 sur " + str(len(valid_pred)) + " données")
    print(confusion_matrix(validation_Y, valid_pred)) 
    print(classification_report(validation_Y, valid_pred))  

    print('debut prediction')
    valid_pred = clf.predict(train_X)
    valid_pred = proba2result(valid_pred).astype('uint8')
    print('nb de 1 : ' + str(valid_pred.sum()))

    # On construit le loader avec 5% des 0
    print("Creation des Loader..")
    train_loader = makeDataLoader(train_X, train_Y, train=True)
    valid_loader = makeDataLoader(validation_X, validation_Y)

    # On construit le modele
    model = BixiNetwork()

    # Load du modele pré entrainé si il existe
    if (os.path.isfile("models/best_model.pth")):
        print("Chargement du modele pré entrainé")
        best_model = load("models/best_model.pth")
    else:
        # Sinon entrainement 
        print("Debut entrainement..")
        best_fscore = 0
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        losses = []
        for epoch in range(1, NB_EPOCH + 1):
            model,loss = train(train_loader, model, optimizer)
            losses.append(loss)

            prediction_epoch = model(from_numpy(np.array(validation_X)))
            prediction_epoch = prediction_epoch.detach().numpy().squeeze()
            prediction_epoch = proba2result(prediction_epoch).astype('uint8')

            f_score = f1_score(validation_Y, prediction_epoch)
            print('fscore neural network : ' + str(f_score))

            accuracy = accuracy_score(validation_Y, prediction_epoch)
            print('accuracy beta neural network : ' + str(accuracy))

            # valid(valid_loader, model)
            if f_score > best_fscore:
                print('mise a jour du meilleur model')
                best_fscore = f_score
                best_model = model
    
    # Sauvegarde du model (bug pour l'instant : supprimer le modele)
    # save(best_model.state_dict(), "models/best_model.pth")




    # On fait les predictions sur l'ensemble de test
    print("Prediction sur l'ensemble de test..")
    print("Best f_score : " + str(best_fscore))
    best_model.eval()

    ### Calcul du fscore


    prediction_train = best_model(from_numpy(np.array(validation_X)))
    prediction_train = prediction_train.detach().numpy().squeeze()
    prediction_train = proba2result(prediction_train).astype('uint8')

    f_score = f1_score(validation_Y, prediction_train)
    print('#####################################################"')
    print('fscore neural network final : ' + str(f_score))

    accuracy = accuracy_score(validation_Y, prediction_train)
    print('accuracy neural network final : ' + str(accuracy))

    print(confusion_matrix(validation_Y, prediction_train)) 
    print(classification_report(validation_Y, prediction_train))  

    prediction = best_model(from_numpy(test_X))
    prediction = prediction.detach().numpy().squeeze()

    # On transforme les probabilités en 0 et 1
    print("Transformation des proba en 0 et 1..")
    prediction = proba2result(prediction).astype('uint8')
    print('nb de 1 dans la prediction : ' + str(prediction.sum()))

    # On écrit les resultats dans le csv de soumission
    print("Ecriture des resultats dans le fichier de soumission..")

    ds.writeCsv(id_test, prediction)

    print("TERMINE")

def train(train_loader, model, optimizer):
    model.train()
    model.double()
    loss_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)  
        loss = F.binary_cross_entropy(output, target.double())
        loss_train = loss_train + loss.item()
        loss.backward()
        optimizer.step()
    return model,loss_train/data.size(0)

def valid(valid_loader, model):
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valid_loss += F.binary_cross_entropy(output, target.double(), size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred).long()).cpu().sum()

    # valid_loss /= len(valid_loader.dataset)
    # print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     valid_loss, correct, len(valid_loader.dataset),
    #     100. * correct / len(valid_loader.dataset)))
    return correct.item() / len(valid_loader.dataset)


def makeDataLoader(X, Y, train=False):
    # On retire beaucoup de 0
    data = []
    ratio = 0.90
    for i in range(len(X)):
        if (train and Y[i] == 0):
            rand = random.random()
            if(rand > ratio):
                data.append([X[i], Y[i]])
        else:
            data.append([X[i], Y[i]])
    print('nb feature reellement utilise : ' + str(len(data)))
    loader = DataLoader(data, shuffle=True, batch_size=1000)
    return loader

def normalize(X, x_min=-1, x_max=1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def proba2result(prediction):
    prediction[prediction <= 0.74] = 0
    prediction[prediction > 0.74] = 1
    return prediction

def preprocessPipeline(data):
    # Remplace les stations par des one hot vector
    # data = np.concatenate((data[:,:-1], get_station_code(data[:,3])), axis=1)

    # Remplace la temperature par un hot vector d'intervalles
    data = np.concatenate((data[:,1:], categorizeTemperatures(data[:,0])), axis=1)
    
    # Meteo NLP
    # data = np.concatenate((data, get_meteo(data[:, 1])), axis=1)
    # np.delete(data, 1, 1)
    
    # TODO Remplace l'heure et le mois par des one hot vector
    data = np.concatenate((data[:,1:], makeHotVector(data[:,0].astype('uint'))), axis=1)
    data = np.concatenate((data[:,1:], get_month(data[:,0].astype('uint'))), axis=1)
    data = np.concatenate((data[:,1:], categorizeDrewPoint(data[:,0])), axis=1)
    data = np.concatenate((data[:,1:], makeHotVector(data[:,0].astype('uint'))), axis=1)

    # data = np.concatenate((data[:,1:], get_station_code(data[:,0].astype('uint16'))), axis=1)

    # TODO Remplace la meteo par des one hot vector
    return data



if __name__ == "__main__":
    main()



