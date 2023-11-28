# David Mathas - November 2023 

# imports:
import numpy as np
import matplotlib.pyplot as plt
from urllib import request
import gzip
import pickle
import os
import math
import random

from help_functions_ML_scripts import sigmoid, softmax, sizes, fnormalize, plot_costs
from code_pbloem_ML_scripts import load_synth, load_mnist
from loop_forward_backward_ML_scripts import forward_pass, backward_pass




def cross_entropy_loss(predicted_probs: list, true_class: int) -> float:
    """
    Calculate the cross-entropy loss.

    Parameters: predicted_probs: list: Predicted probabilities for each class.
                true_class: int: Index of the true class.

    Returns: float: Cross-entropy loss.
    """
    
    #to make sure the the true class index is valid
    assert 0 <= true_class < len(predicted_probs), "Invalid true class index"

    #calculate the negative log probability of the true class
    epsilon = 1e-15 #to ensure non-zero losses
    prob = max(predicted_probs[true_class], epsilon)
    
    return -math.log(prob) 



###### Training the network: ######

#initialization function:

def initialize_parameters(sizes: list) -> dict:
    """
    Initialize the parameters (weights and biases) of the neural network.

    Parameters: sizes: list: Network sizes.

    Returns: dict: Initialized parameters.
    """

    n_input = sizes[0]
    n_hidden = sizes[1]  #nr of nodes in the hidden layer
    n_output = sizes[2]

    #Xavier initialization for weights:
    W = [[random.gauss(0, 0.01) for _ in range(n_hidden)] for _ in range(n_input)]

    #Zero initialization for biases:
    b = [0.0] * n_hidden

    #Xavier initialization for weights:
    V = [[random.gauss(0, 0.01) for _ in range(n_output)] for _ in range(n_hidden)]

    #Zero initialization for biases:
    c = [0.0] * n_output

    #assertion checks:
    assert len(W) == n_input and all(len(row) == n_hidden for row in W)
    assert len(b) == n_hidden
    assert len(V) == n_hidden and all(len(row) == n_output for row in V)
    assert len(c) == n_output

    parameters = {'W': W, 'b': b, 'V': V, 'c': c}

    return parameters


#update (SGD) function:
def update_parameters(parameters: dict, derivatives: dict, 
                      learning_rate: float = 0.01) -> dict:
    """
    Update the parameters using stochastic gradient descent (SGD).

    Parameters: parameters: dict: Current parameters.
                derivatives: dict: Derivatives of the loss with respect to the parameters.
                learning_rate: float: Learning rate for SGD.

    Returns: dict: Updated parameters.
    """

    W = parameters["W"]
    b = parameters["b"]
    V = parameters["V"]
    c = parameters["c"]

    dW = derivatives["dW"]
    db = derivatives["db"]
    dV = derivatives["dV"]
    dc = derivatives["dc"]

    #Update weights for the first layer
    for i in range(len(W)):
        for j in range(len(W[0])):
            W[i][j] -= learning_rate * dW[i][j]

    #Update biases for the first layer
    for i in range(len(b)):
        b[i] -= learning_rate * db[i]

    #Update weights for the second layer
    for i in range(len(V)):
        for j in range(len(V[0])):
            V[i][j] -= learning_rate * dV[i][j]

    #pdate biases for the second layer
    for i in range(len(c)):
        c[i] -= learning_rate * dc[i]

    updated_parameters = {'W': W, 'b': b, 'V': V, 'c': c}

    return updated_parameters



#integregation function (returning list of costs and iteration/epochs):
def train_neural_network(xtrain, ytrain, xval, yval, 
                         sizes: list, 
                         num_epochs: int = 100, 
                         learning_rate: float =0.01) -> list:
    """
    Train the neural network using SGD.

    Parameters: xtrain, ytrain, xval, yval: Training and validation data.
                sizes: list: Network sizes.
                num_epochs: int: Number of training epochs.
                learning_rate: float: Learning rate for SGD.

    Returns: list, list: Lists of training and validation costs for each epoch.
    """
    parameters = initialize_parameters(sizes)
    training_costs = []
    validation_costs = []

    for epoch in range(num_epochs):
        #Training:
        total_cost_train = 0.0
        print('parameters before/after:', parameters)
        for i in range(len(xtrain)):
            input_data = xtrain[i]
            true_class = ytrain[i]

            predicted_probs, cache = forward_pass(input_data, parameters, sizes)
            loss = cross_entropy_loss(predicted_probs, true_class)
            
            derivatives = backward_pass(input_data, parameters, sizes, true_class, cache)
            parameters = update_parameters(parameters, derivatives, learning_rate)

            total_cost_train += loss

        avg_cost_train = total_cost_train / len(xtrain)
        training_costs.append(avg_cost_train)

        #Validation:
        total_cost_val = 0.0
        for i in range(len(xval)):
            input_data_val = xval[i]
            true_class_val = yval[i]

            predicted_probs_val, _ = forward_pass(input_data_val, parameters, sizes)
            loss_val = cross_entropy_loss(predicted_probs_val, true_class_val)

            total_cost_val += loss_val

        avg_cost_val = total_cost_val / len(xval)
        validation_costs.append(avg_cost_val)

        #print loss:
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Cost: {avg_cost_train}, Validation Cost: {avg_cost_val}")

    return training_costs, validation_costs









def main():
    ### Test weights: ###
    print('##########')
    #Define input, output, and weights:
    W = [[1., 1., 1.], [-1., -1., -1.]]
    b = [0., 0., 0.]
    V = [[1., 1.], [-1., -1.], [-1., -1.]]
    c = [0., 0.]
    parameters_1 = {'W': W, 'b': b, 'V': V, 'c': c}
    input_1 = [1., -1.]
    hidden_size = 3
    output_size = 2
    model_1_sizes = sizes(input_1, hidden_size, output_size)

    #Forward pass:
    predicted_probs_model_1, cache_1 = forward_pass(input_1, parameters_1, model_1_sizes)
    print('Predicted probabilities for model 1 and cache:', predicted_probs_model_1, cache_1)
    print('##########')

    #Declare true class:
    true_class_1 = 1
    loss = cross_entropy_loss(predicted_probs_model_1, true_class_1)
    print(f"Cross-entropy loss: {loss}")
    print('##########')

    # Derivatives:
    derivatives_model_1 = backward_pass(input_1, parameters_1, model_1_sizes, true_class_1, cache_1)
    print("Derivatives for the first layer:")
    print("dW:", derivatives_model_1["dW"])
    print("db:", derivatives_model_1["db"])

    print("Derivatives for the second layer:")
    print("dV:", derivatives_model_1["dV"])
    print("dc:", derivatives_model_1["dc"])
    print('##########')

    #Load synthetic data:
    (xtrain, ytrain), (xval, yval), num_cls = load_synth()
    print('Data loaded in.')
    print('Number of classes:', num_cls)
    print('##########')

    xtrain_norm, xval_norm = fnormalize(xtrain), fnormalize(xval)
    print('Data is normalized, e.g., first xtrain set:', xtrain_norm[0], 'becomes:', xtrain_norm[0])
    print('##########')

    print(ytrain)

    parameters = initialize_parameters(model_1_sizes)
    for i in range(25):
        input_data = xtrain[i]
        true_class = ytrain[i]

        predicted_probs, cache = forward_pass(input_data, parameters, model_1_sizes)
        loss = cross_entropy_loss(predicted_probs, true_class)
            
        derivatives = backward_pass(input_data, parameters, model_1_sizes, true_class, cache)
        parameters = update_parameters(parameters, derivatives, 0.01)

        print('pred probs:', predicted_probs)
        print('loss:', loss)
        print('derivatives:', derivatives)
        print('################')
        print('')

    #Initialization:
    print(initialize_parameters(model_1_sizes))

    #running the entire pipeline:
    num_epochs_model = 10
    learning_rate_model = 0.01
    train_cost_1, val_cost_1 = train_neural_network(xtrain_norm, 
                         ytrain, 
                         xval_norm, 
                         yval, 
                         model_1_sizes, num_epochs_model, learning_rate_model)
    
    #plot cots:
    plot_costs(train_cost_1, val_cost_1)


if __name__ == "__main__":
    main()



