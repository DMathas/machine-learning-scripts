# David Mathas - November 2023 

# imports:
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from urllib import request
import gzip
import pickle
import os
import math

from help_functions_ML_scripts import sigmoid, softmax, vec_sizes, fnormalize, plot_costs, normalize_mnist, plot_images_side_by_side
from code_pbloem_ML_scripts import load_synth, load_mnist
from vec_forward_backward_ML_scripts import forward_pass, backward_pass




def cross_entropy_loss(predicted_probs: list, true_class: int) -> float:
    """
    Calculate the cross-entropy loss.

    Parameters: predicted_probs: list: Predicted probabilities for each class.
                rue_class: int: Index of the true class.

    Returns: float: Cross-entropy loss.
    """
    #to make sure the the true class index is valid
    assert 0 <= true_class < len(predicted_probs), "Invalid true class index"

    #calculate the negative log probability of the true class
    true_class = true_class - 1
    epsilon = 1e-15
    prob = max(predicted_probs[true_class], epsilon)
    
    return -math.log(prob)



###### Training the network: ######

#initialization function: 

def initialize_parameters(sizes: list) -> dict:
    """
    Initialize the parameters (weights and biases) of the neural network.

    Parameters:
    sizes: list: Network sizes.

    Returns:
    dict: Initialized parameters.
    """

    n_input, n_hidden, n_output = sizes

    W = np.random.randn(n_input, n_hidden) * 0.01
    b = np.zeros(n_hidden)
    V = np.random.randn(n_hidden, n_output) * 0.01
    c = np.zeros(n_output)

    parameters = {'W': W, 'b': b, 'V': V, 'c': c}

    return parameters


#update (SGD) function:
def update_parameters(parameters: dict, derivatives: dict, 
                      learning_rate: float = 0.2) -> dict:
    """
    Update the parameters using stochastic gradient descent (SGD).

    Parameters: parameters: dict: Current parameters.
                derivatives: dict: Derivatives of the loss with respect to the parameters.
                learning_rate: float: Learning rate for SGD.

    Returns: dict: Updated parameters.
    """

    # for param in parameters:
    #     print(f"param: {param}, derivatives[f'd{param}']: {derivatives[f'd{param}']}")
    #     parameters[param] -= learning_rate * derivatives[f'd{param}'] #to account for the fact that the derivatives are named dW, db, etc....
    
    for param in parameters:
        #Retrieve each parameter and gradient:
        param_value = parameters[param]
        grad_value = derivatives[f'd{param}']
        # print('for:', param)
        # print('param value:', param_value)
        # print('gradient value:', grad_value)

        #update:
        parameters[param] = param_value - learning_rate * np.array(grad_value) #if using numpy
        #parameters[param] = [param_value[i] - learning_rate * grad_value[i] for i in range(len(param_value))]


    return parameters


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
        # print('parameters before/after:', parameters)
        for i in range(len(xtrain)):
            input_data = xtrain[i]
            true_class = ytrain[i]

            predicted_probs, cache = forward_pass(input_data, parameters)
            loss = cross_entropy_loss(predicted_probs, true_class)
            
            derivatives = backward_pass(input_data, parameters, true_class, cache)
            parameters = update_parameters(parameters, derivatives, learning_rate)

            total_cost_train += loss

        avg_cost_train = total_cost_train / len(xtrain)
        training_costs.append(avg_cost_train)

        #Validation:
        total_cost_val = 0.0
        for i in range(len(xval)):
            input_data_val = xval[i]
            true_class_val = yval[i]

            predicted_probs_val, _ = forward_pass(input_data_val, parameters)
            loss_val = cross_entropy_loss(predicted_probs_val, true_class_val)

            total_cost_val += loss_val

        avg_cost_val = total_cost_val / len(xval)
        validation_costs.append(avg_cost_val)

        #Print the cost every 10 iterations
        # if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Cost: {avg_cost_train}, Validation Cost: {avg_cost_val}")

    return parameters, training_costs, validation_costs


def predict_and_compare(image_number: int, 
                            x_data: np.array, 
                            y_data: np.array, 
                            parameters: dict, 
                            sizes: list):
        """
        Predict the class for a given image and compare it to the true class.

        Parameters: image_number: int: Index of the image to predict and compare. 
                    x_data: np.array: Input data.
                    y_data: np.array: True class labels.
                    parameters: dict: Parameters of the neural network.
                    sizes: list: Network sizes.

        Returns: None
        """
        sample_image = x_data[image_number]
        true_class = y_data[image_number]

        #Perform forward pass to get predicted probabilities:
        predicted_probs, _ = forward_pass(sample_image, parameters)
        predicted_class = np.argmax(predicted_probs)

        #Print results:
        print("Predicted Probabilities:", np.round(predicted_probs, 3),
            'True class:', true_class,
            'Predicted class:', predicted_class,
            'Probability:', np.round(predicted_probs[predicted_class], 3))

        #Plot:
        img = x_data[image_number, :].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f'True number {true_class} predicted with {predicted_probs[predicted_class]} prob.')
        plt.show()



def run_experiment_2(xtrain, ytrain, xval, yval, 
                        sizes: list, 
                        num_epochs: int, 
                        learning_rate: float, 
                        num_experiments: int):
        """
        Run multiple experiments with random initialization and plot the average and standard deviation of costs.

        Parameters: xtrain, ytrain, xval, yval: Training and validation data. 
                    sizes: list: Network sizes.
                    num_epochs: int: Number of training epochs.
                    learning_rate: float: Learning rate for SGD.
                    num_experiments: int: Number of experiments to run.

        Returns: None
        """
        #Initialize arrays to store objective values for each experiment:
        all_train_costs = []
        all_val_costs = []

        #Run the experiments:
        for _ in range(num_experiments):
            parameters, train_cost, val_cost = train_neural_network(xtrain, ytrain, xval, yval, sizes, num_epochs, learning_rate)

            all_train_costs.append(train_cost)
            all_val_costs.append(val_cost)

        train_costs = np.array(all_train_costs)
        val_costs = np.array(all_val_costs)

        #Calculate average and standard deviation:
        average_train_cost = np.mean(train_costs, axis=0)
        std_dev_train_cost = np.std(train_costs, axis=0)

        average_val_cost = np.mean(val_costs, axis=0)
        std_dev_val_cost = np.std(val_costs, axis=0)

        #Plot:
        plt.plot(range(1, num_epochs + 1), average_train_cost, label='Average Training Cost', color='blue')
        plt.fill_between(range(1, num_epochs + 1),
                        average_train_cost - std_dev_train_cost,
                        average_train_cost + std_dev_train_cost,
                        facecolor='blue', alpha=0.8, label='Training Standard Deviation')

        plt.plot(range(1, num_epochs + 1), average_val_cost, label='Average Validation Cost', color='orange')
        plt.fill_between(range(1, num_epochs + 1),
                        average_val_cost - std_dev_val_cost,
                        average_val_cost + std_dev_val_cost,
                        facecolor='orange', alpha=0.8, label='Validation Standard Deviation')

        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Average and Standard Deviation of Training and Validation Costs')
        plt.legend()
        plt.grid(True)
        plt.show()


def calculate_accuracy(parameters, sizes, x_data, true_labels):
    """
    Calculate the accuracy of the model on the given data.

    Parameters: parameters: dict: Dictionary containing the parameters of the neural network.  
                sizes: list: Network sizes.
                x_data: np.array: Input data.
                true_labels: list: True labels for the input data.

    Returns: float: Accuracy percentage.
    """

    predicted_labels = []

    #Loop over all images in the data:
    for i in range(len(x_data)):

        sample_image = x_data[i]
        true_class = true_labels[i]

        #get predicted probabilities:
        predicted_probs, _ = forward_pass(sample_image, parameters)
        # print('pred prob.:', predicted_probs)
        #Get the predicted class:
        predicted_class = np.argmax(predicted_probs)

        predicted_labels.append(predicted_class + 1)


    correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))

    #Calculate accuracy:
    accuracy = (correct_predictions / len(true_labels)) * 100

    return accuracy


def evaluate_learning_rates(learning_rates: list, 
                            x_train: np.array, 
                            y_train: list, 
                            x_val: np.array, 
                            y_val: list, 
                            sizes: list, 
                            num_epochs: int):
    """
    Evaluate different learning rates on the neural network and calculate validation accuracies.

    Parameters: learning_rates: list: List of learning rates to evaluate. 
                x_train: np.array: Training input data.
                y_train: list: Training true labels.
                x_val: np.array: Validation input data.
                y_val: list: Validation true labels.
                sizes: list: Network sizes.
                num_epochs: int: Number of training epochs.

    Returns: dict: A dictionary where keys are learning rates (as strings) and values are corresponding validation accuracies.
    """
    val_accuracies = {}

    for learning_rate in learning_rates:
        #Train the neural network with current learning rate:
        parameters, _, _ = train_neural_network(x_train, y_train, x_val, y_val, sizes, num_epochs, learning_rate)

        #calculate validation accuracy:
        accuracy = calculate_accuracy(parameters, sizes, x_val, y_val)

        #Store the accuracy in the dictionary:
        val_accuracies[str(learning_rate)] = accuracy

        print(f"Validation Accuracy (Learning Rate {learning_rate}): {accuracy:.2f}%")

    return val_accuracies







def main():
    
    ################### 
    #MNIST data vectorized neural network:

    #load MNIST data
    (xtrain_mnist, ytrain_mnist), (xval_mnist, yval_mnist), num_cls_mnist = load_mnist()

    

    #normalize pixel values by greyscale range:
    xtrain_mnist_norm, xval_mnist_norm = normalize_mnist(xtrain_mnist), normalize_mnist(xval_mnist)
    
    #plot image (code from repo of data loader on github) original vs normalized:
    # plot_images_side_by_side(xtrain_mnist, xtrain_mnist_norm, 2)

    print('length simple train set:', len(xtrain_mnist_norm))

    hidden_size = 300
    output_size = 10
    model_2_sizes = vec_sizes(xtrain_mnist, hidden_size, num_cls_mnist)

    # running the entire pipeline:
    num_epochs_model = 5
    learning_rate_model = 0.01
    parameters, train_cost_1, val_cost_1 = train_neural_network(xtrain_mnist_norm, 
                         ytrain_mnist, 
                         xval_mnist_norm, 
                         yval_mnist, 
                         model_2_sizes, num_epochs_model, learning_rate_model)





    # ### Running some experiments ###
    #Experiment 1: #in overleaf
    # #plot cots:
    plot_costs(train_cost_1, val_cost_1)

    #Experiment 2: 
    num_epochs_model = 5
    # learning_rate_model = 0.01
    # num_experiments = 3
    # run_experiment_2(xtrain_mnist_norm, ytrain_mnist, xval_mnist_norm, yval_mnist, model_2_sizes, num_epochs_model, learning_rate_model, num_experiments)


    #Experiment 3: accuracy as performance measure 
    #caculate accuracies for different learning rates:
    learning_rates_e3 = [0.001, 0.003, 0.01, 0.03, 0.1]
    results = evaluate_learning_rates(learning_rates_e3, 
                                      xtrain_mnist_norm, 
                                      ytrain_mnist, 
                                      xval_mnist_norm, 
                                      yval_mnist, 
                                      model_2_sizes, 
                                      num_epochs_model)
    #print rsults:
    print("Validation Accuracies:")
    for rate, accuracy in results.items():
        print(f"Learning Rate {rate}: {accuracy:.2f}%")
    #result --> optimal learning rate seems to be 0.03



    # #Experiment 4 to show result in cannonical set...:
    (xtrain_mnist_can, ytrain_mnist_can), (xtest_mnist_can, ytest_mnist_can), num_cls_mnist = load_mnist(final=True)
    print('length canonical train set:', len(xtrain_mnist_can))

    #normalize pixel values by greyscale range:
    xtrain_mnist_can_norm, xtest_mnist_can_norm = normalize_mnist(xtrain_mnist_can), normalize_mnist(xtest_mnist_can)
    learning_rate_optim = 0.03

    #training full model 
    parameters_can, train_cost_can, test_cost_can = train_neural_network(xtrain_mnist_can_norm, 
                         ytrain_mnist_can, 
                         xtest_mnist_can_norm, 
                         ytest_mnist_can, 
                         model_2_sizes, num_epochs_model, learning_rate_optim)
    #plot loss curves:
    plot_costs(train_cost_can, test_cost_can)
    #calculate accuracy after traoining on canonical set:
    accuracy_can = calculate_accuracy(parameters_can, model_2_sizes, xtest_mnist_can_norm, ytest_mnist_can)
    print('Test set accuracy is:', accuracy_can)

    # #check two pictures:
    predict_and_compare(0, xtest_mnist_can_norm, ytest_mnist_can, parameters_can, model_2_sizes)
    predict_and_compare(3, xtest_mnist_can_norm, ytest_mnist_can, parameters_can, model_2_sizes)
    

if __name__ == "__main__":
    main()
