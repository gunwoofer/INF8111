import torch
from torch.autograd import Variable     
import torch.nn as nn 
import bixi_dataset as ds
import bixi_network as net
import utils_train as utils


def main():
    print("TP2..")
    cuda = torch.device('cuda') 
    #### CREATION DES MATRICE DE FEATURES/TARGET EN TRAIN, VALIDATION ET TEST
    train_X, train_Y = ds.getData(ds.TRAIN_FILE_NAME)
    index = int(0.6 * train_X.shape[0])
    validation_X, validation_Y = train_X[index:], train_Y[index:]
    train_X, train_Y = train_X[:index], train_Y[:index]
    test = ds.getData(ds.TEST_FILE_NAME, type_data='test')

    batch_size = 100
    test_batch_size = 100

    train_X = torch.from_numpy(train_X)#changer le device si vous utilisez pas cuda
    train_Y = torch.from_numpy(train_Y)# changer le device si vous utilisez pas cuda
    dataset = torch.utils.data.TensorDataset(train_X, train_Y)

    train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader((validation_X, validation_Y),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test,
        batch_size=batch_size, shuffle=True)

    print('chargement des données fini')
    
    
    
    ########## PYTORCH #################




    print('debut des experimentations')
    best_precision = 0
    for model in [net.BixiNetwork()]:  # Liste des modeles que l'on veut utiliser pour tester. Mettez à jour pour éviter de perdre du temps
        #model.cuda()  # Commenter cette ligne si vous avez pas cuda
        model.double()
        model, precision = utils.experiment(model, train_loader, valid_loader)
        if precision > best_precision:
            best_precision = precision
            best_model = model
            print('nouveau model detecte meilleur avec une precision de : ' + best_precision)


if __name__ == "__main__":
    main()