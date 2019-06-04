import bixi_dataset as ds
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from torch import from_numpy
from torch import mean, std
from preprocess import get_station_code
from bixi_network import BixiNetwork

LEARNING_RATE = 0.01
NB_EPOCH = 5

def main():
    # On recupere les donnees d'entrainement
    print("Recuperation des donnees d'entrainement..")
    train_X, train_Y, id_train = ds.getData(ds.TRAIN_FILE_NAME)

    # On recupere les donnees de test a predire
    print("Recuperation des donnees de test..")
    test_X, _, id_test = ds.getData(ds.TEST_FILE_NAME, type_data='test')

    # On divise en set de validation et d'entrainement
    print("Division du set d'entrainement en validation-entrainement..")
    index = int(0.6 * train_X.shape[0])
    validation_X, validation_Y = train_X[index:], train_Y[index:]
    train_X, train_Y = train_X[:index], train_Y[:index]

    # On normalise les données
    print("Normalisation des données..")
    train_X = normalize(train_X)
    validation_X = normalize(validation_X)
    test_X = normalize(test_X)

    # On construit le loader avec 5% des 0
    print("Creation des Loader..")
    train_loader = makeDataLoader(train_X, train_Y, train=True)
    valid_loader = makeDataLoader(validation_X, validation_Y)

    # On construit le modele
    model = BixiNetwork()

    # Entrainement
    print("Debut entrainement..")
    best_precision = 0
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    for epoch in range(1, NB_EPOCH + 1):
        model,loss = train(train_loader, model, optimizer)
        losses.append(loss)
        precision = valid(valid_loader, model)
        if precision > best_precision:
            best_precision = precision
            best_model = model
    
    # On fait les predictions sur l'ensemble de test
    print("Prediction sur l'ensemble de test..")
    prediction = best_model(from_numpy(test_X))

    # breakpoint
    test = 2

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

    valid_loss /= len(valid_loader.dataset)
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return correct.item() / len(valid_loader.dataset)


def makeDataLoader(X, Y, train=False):
    # On retire beaucoup de 0
    data = []
    ratio = 0.95
    for i in range(len(X)):
        if (train and Y[i] == 0):
            rand = random.random()
            if(rand > ratio):
                data.append([X[i], Y[i]])
        else:
            data.append([X[i], Y[i]])
    loader = DataLoader(data, shuffle=True, batch_size=100)
    return loader

def normalize(X, x_min=-1, x_max=1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 


if __name__ == "__main__":
    main()