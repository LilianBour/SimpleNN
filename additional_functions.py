import h5py
import numpy as np
from NN_module import L_model_forward
import matplotlib.pyplot as plt

#Load catvsnoncat dataset
def load_dataset():
    train_dataset = h5py.File('Data\\catvsnoncat\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('Data\\catvsnoncat\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments
    X : data set of examples you would like to label
    parameters : parameters of the trained model

    Returns
    p : predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  #number of layers in the neural network
    p = np.zeros((1, m))

    #Forward propagation
    probas, caches = L_model_forward(X, parameters)

    #Convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))
    return p

def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()