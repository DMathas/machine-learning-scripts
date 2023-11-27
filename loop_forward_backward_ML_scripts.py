#contains loop-based forward and backward functions without using numpy (only using math package)

from help_functions_ML_scripts import sigmoid, softmax

def forward_pass(input: list, 
                 parameters: dict, 
                 sizes: list):
    """
    Perform the forward pass of the neural network using looped operations.

    Parameters: input_data: list: Input data.
                parameters: dict: Dictionary containing the parameters of the neural network.
                sizes: list: List containing the sizes of the input layer, hidden layer, and output layer.

    Returns: tuple: A tuple containing the predicted probabilities and the cache for later use in backpropagation.
    """

    #obtain parameters:
    W = parameters['W']
    b = parameters['b']
    V = parameters['V']
    c = parameters['c']

    #Initialize vectors of zeros based on sizes:
    k = [0.0] * sizes[1]  #linear outputs in the hidden layer
    h = [0.0] * sizes[1]  #activated neurons
    output = [0.0] * sizes[2]
    
    #loop for linear output
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            k[j] += W[i][j] * input[i] 
        k[j] += b[j]

    assert (len(k) == sizes[1])

    #activate the neurons:
    for i in range(sizes[1]):
        h[i] = sigmoid(k[i])

    #linear output:
    for i in range(sizes[1]):
        for j in range(sizes[2]):
            output[j] += V[i][j] * h[i]
        output[j] += c[j]

    predicted_probs = softmax(output)

    #dictionary to save the linear values and activated neurons in hidden layer:
    cache = {"k": k,
             "h": h,
             "output": output,
             "probs": predicted_probs}

    return predicted_probs, cache


## BACKWARD PASS: ##

# def backward_pass(input: list, 
#                   parameters: list,
#                   sizes: list,
#                   true_class: int, 
#                   cache: dict) -> dict:
#     """
#     Perform one backward pass and calculate derivatives.

#     Parameters: input: list: Input values.
#                 W: list: First layer weights.
#                 b: list: First layer biases.
#                 V: list: Second layer weights.
#                 c: list: Second layer biases.
#                 sizes: list: Network sizes.
#                 true_class: int: Index of the true class.
#                 cache: dict: Cached values from the forward pass.

#     Returns: dict: Derivatives of the loss with respect to the parameters.
#     """

#     #obtain parameters:
#     W = parameters['W']
#     b = parameters['b']
#     V = parameters['V']
#     c = parameters['c']

#     true_class = true_class - 1

#     #Unpack cache values from forward propagation:
#     k = cache["k"]
#     h = cache["h"]
#     output = cache["output"]
#     out_probs = cache["probs"]


#     dW = [[0.0 for _ in range(sizes[1])] for _ in range(sizes[0])]
#     db = [0.0 for _ in range(sizes[1])]
#     dV = [[0.0 for _ in range(sizes[2])] for _ in range(sizes[1])]
#     dk = [[0.0 for _ in range(sizes[2])] for _ in range(sizes[1])]
#     dh = [[0.0 for _ in range(sizes[2])] for _ in range(sizes[1])]
#     dc = [0.0 for _ in range(sizes[2])]

#     #Backprop for second layer:
#     for i in range(sizes[1]):
#         for j in range(sizes[2]):
#             dc[j] = out_probs[j] - (1 if j == true_class else 0)
#             dV[i][j] = h[i] * (out_probs[j] - (1 if j == true_class else 0))

#             #Derivative of the loss with respect to the linear output of the second layer
#             dh[i][j] = V[i][j] * (out_probs[j] - (1 if j == true_class else 0)) * h[i] * (1 - h[i])

#     #Derivative of the loss with respect to the hidden layer activations
#     dk = [sum(dk[i][j] * V[i][j] for j in range(sizes[2])) for i in range(sizes[1])]
#     dk = [round(value, 10) for value in dk]
    
#     #Backproo for first layer:
#     # print(input)
#     # print('dk is:', dk)
#     for i in range(sizes[0]):
#         for j in range(sizes[1]):
#             dW[i][j] = dk[j] * abs(input[i]) #assuming the data is normalized to be between 0 and 1

#     db = dk

#     derivatives = {"dW": dW, "db": db, "dV": dV, "dc": dc}

#     return derivatives




def backward_pass(input: list, 
                  parameters: list,
                  sizes: list,
                  true_class: int, 
                  cache: dict) -> dict:
    """
    Perform one backward pass and calculate derivatives.

    Parameters: input: list: Input values.
                W: list: First layer weights.
                b: list: First layer biases.
                V: list: Second layer weights.
                c: list: Second layer biases.
                sizes: list: Network sizes.
                true_class: int: Index of the true class.
                cache: dict: Cached values from the forward pass.

    Returns: dict: Derivatives of the loss with respect to the parameters.
    """

    #obtain parameters:
    W = parameters['W']
    b = parameters['b']
    V = parameters['V']
    c = parameters['c']

    true_class = true_class - 1

    #Unpack cache values from forward propagation:
    k = cache["k"]
    h = cache["h"]
    output = cache["output"]
    out_probs = cache["probs"]

    n_output = sizes[2]
    n_hidden = sizes[1]
    n_input = sizes[0]

    dy = [0.0 for _ in range(n_output)]
    do = [0.0 for _ in range(n_output)]
    dh = [0.0 for _ in range(n_hidden)]
    dk = [0.0 for _ in range(n_hidden)]
    dV = [[0.0 for _ in range(n_output)] for _ in range(n_hidden)]
    dW = [[0.0 for _ in range(n_hidden)] for _ in range(n_input)]
    

# the loop under this should give the following:
# do[0] = dy[0] * out_probs[0] * (1 - out_probs[0]) + dy[1] * - out_probs[0] * out_probs[1]
# do[1] = dy[1] * out_probs[1] * (1 - out_probs[1]) + dy[0] * - out_probs[1] * out_probs[0]

    #Backprop for output and second layer:
    for j in range(n_output):
        dy[j] = -1 / out_probs[j] * (1 if j == true_class else 0)
        
        for i in range(n_output):
            do[j] += (dy[i] * out_probs[i] * (1 - out_probs[j])) if i == j else (-dy[i] * out_probs[i] * out_probs[j]) #MVCR
    # print(dy)
    dc = do
    # print('dy:', dy)
    # print('do:', dc)

    # print(do[0], V[0][0])
    # print(do[1], V[0][1])

    #USE MVC rule! #for dh, this loop: --> Using multivar chain rule:
    for i in range(n_hidden):
        for j in range(n_output):
            dV[i][j] = do[j] * h[i]
            dh[i] += do[j] * V[i][j] #MVCR
        dk[i] = dh[i] * h[i] * (1 - h[i])
    # print('dv:', dV)

    
    # print('dk:', dk)    # SHOULD OUTPUT THIS:
    # dh[0] = do[0] * V[0][0] + do[1] * V[0][1]
    # dh[1] = do[0] * V[1][0] + do[1] * V[1][1]
    # dh[2] = do[0] * V[2][0] + do[1] * V[2][1]
    # print('dh:', dh)
    #Backprop for hidden layer:
    for i in range(n_input):
        for j in range(n_hidden):
            dW[i][j] = dk[j] * input[i]
    # print(dW)
    db = dk

    derivatives = {"dW": dW, "db": db, "dV": dV, "dc": dc}

    return derivatives




# #Backprop for second layer:
#     for i in range(sizes[1]):
#         for j in range(sizes[2]):
#             dc[j] = out_probs[j] - (1 if j == true_class else 0)
#             dV[i][j] = h[i] * (out_probs[j] - (1 if j == true_class else 0))

#             #Derivative of the loss with respect to the linear output of the second layer
#             dh[i][j] = V[i][j] * (out_probs[j] - (1 if j == true_class else 0)) * h[i] * (1 - h[i])

#     #Derivative of the loss with respect to the hidden layer activations
#     dk = [sum(dk[i][j] * V[i][j] for j in range(sizes[2])) for i in range(sizes[1])]
#     dk = [round(value, 10) for value in dk]
    
#     #Backproo for first layer:
#     # print(input)
#     # print('dk is:', dk)
#     for i in range(sizes[0]):
#         for j in range(sizes[1]):
#             dW[i][j] = dk[j] * abs(input[i]) #assuming the data is normalized to be between 0 and 1

#     db = dk