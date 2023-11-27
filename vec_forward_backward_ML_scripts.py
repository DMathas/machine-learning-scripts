#contains vectorized forward and backward functions:

import numpy as np
from help_functions_ML_scripts import vec_sigmoid, vec_softmax

def forward_pass(input: np.array, parameters: dict):
    """
    Perform the forward pass of the neural network.

    Parameters:
    input_data: np.array: Input data.
    parameters: dict: Dictionary containing the parameters of the neural network.

    Returns: tuple: A tuple containing the predicted probabilities and the cache for later use in backpropagation.
    """
    W = parameters['W']
    b = parameters['b']
    V = parameters['V']
    c = parameters['c']

    k = W.T @ input + b
    h = vec_sigmoid(k)
    output = V.T @ h + c

    predicted_probs = vec_softmax(output)

    cache = {"k": k, "h": h, "output": output, "probs": predicted_probs}

    return predicted_probs, cache

def backward_pass(input: np.array, parameters: dict, true_class: int, cache: dict):
    """
    Perform the backward pass of the neural network to compute derivatives.

    Parameters:
    input_data: np.array: Input data.
    parameters: dict: Dictionary containing the parameters of the neural network.
    true_class: int: Index of the true class.
    cache: dict: Dictionary containing cached intermediate values from the forward pass.

    Returns:
    dict: Dictionary containing the derivatives of the parameters with respect to the loss.
    """
    W = parameters['W']
    b = parameters['b']
    V = parameters['V']
    c = parameters['c']

    #adjust true class for python indexing:
    true_class = true_class - 1

    k = cache["k"]
    h = cache["h"]
    output = cache["output"]
    out_probs = cache["probs"]

    #Backprop for the second layer (directly to dc and onwards):
    dc = out_probs.copy()
    dc[true_class] -= 1
    dV = np.outer(h, dc) 

    #Backprop for hidden layer:  
    dh = V @ dc
    dk = dh * h * (1 - h)

    dW = np.outer(input, dk) 
    db = dk

    derivatives = {"dW": dW, "db": db, "dV": dV, "dc": dc}

    return derivatives





# RARE FUNCTIE cost blijft constant...
# def backward_pass(input: np.array, parameters: dict, true_class: int, cache: dict) -> dict:
#     """
#     Perform one backward pass and calculate derivatives.

#     Parameters: input: np.array: Input values.
#                 parameters: dict: Dictionary containing the parameters of the neural network.
#                 sizes: list: Network sizes.
#                 true_class: int: Index of the true class.
#                 cache: dict: Cached values from the forward pass.

#     Returns: dict: Derivatives of the loss with respect to the parameters.
#     """
#     # Obtain parameters:
#     W = parameters['W']
#     b = parameters['b']
#     V = parameters['V']
#     c = parameters['c']

#     true_class = true_class - 1

#     # Unpack cache values from forward propagation:
#     k = cache["k"]
#     h = cache["h"]
#     output = cache["output"]
#     out_probs = cache["probs"]

#     # Initialize derivatives with zeros:
#     dW = np.zeros_like(W)
#     db = np.zeros_like(b)
#     dV = np.zeros_like(V)
#     dc = np.zeros_like(c)

#     # Backprop for output and second layer:
#     dy = -1 / out_probs
#     dy[true_class] += 1
#     do = dy * out_probs * (1 - out_probs)

#     # Update dc:
#     dc = do

#     # Use MVC rule for dV:
#     dV = np.outer(h, do)

#     # Use MVC rule for dh:
#     dh = V @ do

#     # Backprop for hidden layer:
#     dk = dh * h * (1 - h)
#     dW = np.outer(input, dk)
#     db = dk

#     derivatives = {"dW": dW, "db": db, "dV": dV, "dc": dc}

#     return derivatives

