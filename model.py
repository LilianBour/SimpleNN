import numpy as np
from NN_module import initalize_param_deep, L_model_forward, L_model_backward, update_parameters, get_cost
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments
    X : data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y : true "label" vector of shape (1, number of examples)
    layers_dims : list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate : learning rate of the gradient descent update rule
    num_iterations : number of iterations of the optimization loop
    print_cost : if True, it prints the cost every 100 steps

    Returns
    parameters : parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  #keep track of cost

    # Parameters initialization.
    parameters = initalize_param_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        cost = get_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        #Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs