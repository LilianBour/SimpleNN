from model import L_layer_model
import h5py
import numpy as np
import matplotlib.pyplot as plt
from additional_functions import load_dataset, predict, plot_costs

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

learning_rate = 0.004375
num_iterations = 3750
layers_dims = [12288,20, 7, 5, 1]
parameters, costs = L_layer_model(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost = True)
plot_costs(costs, learning_rate)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
