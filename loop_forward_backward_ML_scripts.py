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

    #Backprop for output and second layer:
    for j in range(n_output):
        dy[j] = -1 / out_probs[j] * (1 if j == true_class else 0)
        
        for i in range(n_output):
            do[j] += (dy[i] * out_probs[i] * (1 - out_probs[j])) if i == j else (-dy[i] * out_probs[i] * out_probs[j]) #MVCR
    dc = do
                    
    #USE MVC rule!
    for i in range(n_hidden):
        for j in range(n_output):
            dV[i][j] = do[j] * h[i]
            dh[i] += do[j] * V[i][j] #MVCR
        dk[i] = dh[i] * h[i] * (1 - h[i])

    #Backprop for hidden layer:
    for i in range(n_input):
        for j in range(n_hidden):
            dW[i][j] = dk[j] * input[i]
    # print(dW)
    db = dk

    derivatives = {"dW": dW, "db": db, "dV": dV, "dc": dc}

    return derivatives
