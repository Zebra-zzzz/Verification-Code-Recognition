import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def load_datasets():
    train_dataset = h5py.File('./datasets/codes.hdf5', "r")
    train_set_x_orig = np.array(train_dataset["train_images"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_codename"][:]) # your train set labels


    test_dataset = h5py.File('datasets/codes.hdf5', "r")
    test_set_x_orig = np.array(test_dataset["test_images"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_codename"][:]) # your test set labels

    # classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    # train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    # test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    # return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

if __name__ == '__main__':
    x, y,x1,y1 = load_datasets()
    print(y[2])