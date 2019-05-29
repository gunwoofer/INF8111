import torch
import bixi_dataset as ds




def main():
    print("TP2..")

    #### CREATION DES MATRICE DE FEATURES/TARGET EN TRAIN, VALIDATION ET TEST
    train_X, train_Y = ds.getData(ds.TRAIN_FILE_NAME)
    index = int(0.6 * train_X.shape[0])
    validation_X, validation_Y = train_X[index:], train_Y[index:]
    train_X, train_Y = train_X[:index], train_Y[:index]
    test_X, _ = ds.getData(ds.TEST_FILE_NAME, type_data='test')

    ########## PYTORCH #################


if __name__ == "__main__":
    main()