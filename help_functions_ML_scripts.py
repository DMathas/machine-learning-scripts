import math
import matplotlib.pyplot as plt
import numpy as np

# from vec_forward_backward_DL_A1 import forward_pass

#define some helper functions: 
def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid activation function.

    Parameters: x: float: Input value.

    Returns: float: calculated sigmoid of the input value.
    """

    return 1 / (1 + math.exp(-x))

#vecorized sigmoid:
def vec_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Calculate the sigmoid activation function for an array.

    Parameters: x: np.ndarray: Input values.

    Returns: np.ndarray: Calculated sigmoid of the input values.
    """

    return 1 / (1 + np.exp(-x))


def softmax(x: list) -> list:
    """
    Calculate the softmax activation function for an array (list).

    Parameters: x: list: Input values

    Returns: list: Softmax of the input values.
    """
    
    max_val = max(x)
    # exp_values = [math.exp(xi - max_val) for xi in x] #for more stability 
    exp_values = [math.exp(xi) for xi in x]
    exp_sum = sum(exp_values)
    return [exp_i / exp_sum for exp_i in exp_values]

#vectorized softmax:
def vec_softmax(x: np.ndarray) -> np.ndarray:
    """
    Calculate the softmax activation function for an array.

    Parameters: x: np.ndarray: Input values.

    Returns: np.ndarray: Softmax of the input values.
    """
    max_val = np.max(x)
    exp_values = np.exp(x - max_val)
    exp_sum = np.sum(exp_values)
    return exp_values / exp_sum


def sizes(input: np.array, hidden: int, output: int):
    """
    Calculate the sizes of different layers in the neural network.

    Parameters: input_data: np.array: Input data.
                hidden: int: Number of nodes in the hidden layer.
                output: int: Number of nodes in the output layer.

    Returns: list: List containing the sizes of the input layer, hidden layer, and output layer.
    """

    n_input = len(input)
    n_hidden = hidden #number of nodes in the hidden layer
    n_output = output

    return [n_input, n_hidden, n_output]

#vectorized sizes version:
def vec_sizes(input: np.array, hidden: int, output: int):
    """
    Calculate the sizes of different layers in the neural network using vectorized operations.

    Parameters: input_data: np.array: Input data.
                hidden: int: Number of nodes in the hidden layer.
                output: int: Number of nodes in the output layer.

    Returns: list: List containing the sizes of the input layer, hidden layer, and output layer.
    """

    n_input = input.shape[1]
    n_hidden = hidden #number of nodes in the hidden layer
    n_output = output

    return [n_input, n_hidden, n_output]

def fnormalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize the input data.

    Parameters: data: np.ndarray: Input data to be normalized.

    Returns: np.ndarray: Normalized data.
    """

    #calculate the minimum and maximum values along each column:
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    #calculate the range of values:
    range_vals = max_vals - min_vals

    #avoid division by zero:
    range_vals[range_vals == 0] = 1.0

    #normalize the data to be between 0 and 1:
    normalized_data = (data - min_vals) / range_vals

    return normalized_data
    

# def fnormalize(data: np.ndarray) -> np.ndarray:
#     """
#     Normalize the input data using tanh function.

#     Parameters: data: np.ndarray: Input data to be normalized.

#     Returns: np.ndarray: Normalized data.
#     """
#     # Apply tanh normalization to each column
#     normalized_data = np.tanh(data)

#     return normalized_data


def plot_costs(train_costs: list, val_costs: list):
    """
    Plot training and validation costs over epochs.

    Parameters: train_costs: list: List of training costs for each epoch. 
                val_costs: list: List of validation costs for each epoch.
    """

    epochs = range(1, len(train_costs) + 1)

    plt.plot(epochs, train_costs, label='Training Cost')
    plt.plot(epochs, val_costs, label='Validation Cost')

    plt.title('Training and Validation Costs')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()


def normalize_mnist(images: np.array) -> np.array:
    """
    Normalize the pixel values of MNIST images to the range [0, 1].
                    
    Parameters: images: np.array: Input MNIST images.

    Returns: normalized_images: np.array: Normalized images.
        """

    normalized_images = images / 255.0

    return normalized_images


def plot_images_side_by_side(original_data: np.array, 
                             normalized_data: np.array, 
                             img_nr: int) -> None:
    """
    Plot the original and normalized images side by side. Alternatively, can also be used to plot two images and their predicted classes. 

    Parameters: original_data:  np.array, original image data 
                normalized_data:  np.array, normalized image data
                img_nr: int: image number to display
    """

    #Extract images from the data:
    img_original = original_data[img_nr, :].reshape(28, 28)
    img_normalized = normalized_data[img_nr, :].reshape(28, 28)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_normalized, cmap='gray')
    plt.title('Normalized Image')

    plt.show()

# def calculate_accuracy(predicted_labels, true_labels):
#         """DOCSTRINFS"""

#         predicted_probs, _ = forward_pass(sample_image, parameters, sizes)
#         predicted_class = np.argmax(predicted_probs)

#         #Count correct predictions:
#         correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))

#         #Calculate accuracy:
#         accuracy = (correct_predictions / len(true_labels)) * 100

#         return accuracy






# def plot_images_side_by_side(image1, image2, title1="", title2=""):
#     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#     axs[0].imshow(image1, cmap='gray')
#     axs[0].set_title(title1)
#     axs[1].imshow(image2, cmap='gray')
#     axs[1].set_title(title2)
#     plt.show()


