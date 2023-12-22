import struct
import random
import math
import numpy as np
import matplotlib.pyplot as plt

def read_labels(filename):
    with open(filename, 'rb') as file:
        magic, num_items = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Invalid magic number in MNIST label file')
        labels = file.read(num_items)
    return np.array([label for label in labels], dtype=np.uint8)

def read_images(filename):
    with open(filename, 'rb') as file:
        magic, num_items, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Invalid magic number in MNIST image file')
        images = np.fromfile(file, dtype=np.uint8)
        images = images.reshape(num_items, rows, cols)
    return images

def get_random_pattern(images, labels):
    index = random.randint(0, len(images) - 1)
    return labels[index], index

def set_weights(num_inputs, num_neurons):
    weights = initialize_weights(num_inputs, num_neurons)
    normalized_weights = normalize_weights(weights)
    return normalized_weights

def initialize_weights(num_inputs, num_neurons):
    return [[random.uniform(-1, 1) for _ in range(num_inputs)] for _ in range(num_neurons)]

def normalize_weights(weights):
    normalized_weights = []
    for neuron_weights in weights:
        weight_sum = sum(neuron_weights)
        # Avoid division by zero
        if weight_sum == 0:
            normalized_neuron_weights = neuron_weights
        else:
            normalized_neuron_weights = [weight / weight_sum for weight in neuron_weights]
        normalized_weights.append(normalized_neuron_weights)
    return normalized_weights

def propagate_on_retina(image):
    return image.flatten() / 255.0

# Calculate X_h
def calculate_hidden_layer_output(input_vector, weights_input_hidden):
    hidden_layer_output = []
    hidden_neuron = range(len(weights_input_hidden))
    input_neuron = range(len(input_vector))
    
    for j in hidden_neuron: 
        pot_h = 0
        for i in input_neuron:  
            pot_h += input_vector[i] * weights_input_hidden[j][i]
        hidden_layer_output.append(sigmoid(pot_h))
    return hidden_layer_output, pot_h

# Calculate X_i
def calculate_output_layer_output(hidden_layer_output, weights_hidden_output):
    output_layer_output = []
    output_neuron = range(len(weights_hidden_output))
    hidden_neuron = range(len(hidden_layer_output))
    
    for k in output_neuron:  
        pot_i = 0
        for j in hidden_neuron:  
            pot_i += hidden_layer_output[j] * weights_hidden_output[k][j]
        output_layer_output.append(sigmoid(pot_i))
    return output_layer_output, pot_i

def sigmoid(x):
    """Compute the sigmoid function safely for large x."""
    if x < -700:  # Values less than -700 will cause overflow in `exp`.
        return 0
    elif x > 700:  # Values greater than 700 will lead to a sigmoid very close to 1.
        return 1
    return 1 / (1 + math.exp(-x))

def calculate_delta_output_layer(pot_i, label, x_i):
    delta = []
    for i in range(len(x_i)):
        # Calculate the error term for the output neuron
        # Correct class is 1, rest are 0s
        expected_value = 1.0 if i == label else 0.0
        delta.append(sigmoid_derivative(pot_i) * (expected_value - x_i[i]))
    return delta

def calculate_delta_hidden_layer(pot_h, delta_output, weights_hidden_output, x_h):
    delta_hidden = []
    for j in range(len(x_h)):
        # Calculate the error term for the hidden neuron
        error_term = sum(delta_output[k] * weights_hidden_output[k][j] for k in range(len(delta_output)))
        delta_hidden.append(sigmoid_derivative(pot_h) * error_term)
    return delta_hidden

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def learn(weights, delta_i, epsilon, x_j):
    for i in range(len(weights)):
        weights[i] += epsilon * delta_i * x_j[i]
    return weights

def calculate_error_percentage(delta_output, p):
    error_sum = np.sum(np.abs(delta_output))
    error_percentage = (error_sum / len(delta_output)) * 100
    return error_percentage / p

def launch_network(labelsFile, imagesFile):
    input_size = 28 * 28
    hidden_size = 100
    output_size = 10
    threshold = 0.04
    epsilon = 0.01 # Learning rate
    
    error_percentages = [] # plotting purpose
    error_percentage = float("inf")
    while error_percentage > threshold:
      
        weights_input_hidden = set_weights(input_size, hidden_size)
        weights_hidden_output  = set_weights(hidden_size, output_size)

        labels = read_labels(labelsFile)
        images = read_images(imagesFile)
        label, image_index = get_random_pattern(images, labels)

        x_j = propagate_on_retina(images[image_index])
        x_h, pot_h = calculate_hidden_layer_output(x_j, weights_input_hidden)
        x_i, pot_i = calculate_output_layer_output(x_h, weights_hidden_output)

        delta_i = calculate_delta_output_layer(pot_i, label, x_i)
        delta_h = calculate_delta_hidden_layer(pot_h, delta_i, weights_hidden_output, x_h)

        # Update the weights for the hidden-output layer
        for k in range(output_size):
            weights_hidden_output[k] = learn(weights_hidden_output[k], delta_i[k], epsilon, x_h)

        # Update the weights for the input-hidden layer
        for j in range(hidden_size):
            weights_input_hidden[j] = learn(weights_input_hidden[j], delta_h[j], epsilon, x_j)

        error_percentage = calculate_error_percentage(delta_i, 100) 
        error_percentages.append(error_percentage)  
        print(f'Current error percentage: {error_percentage:.2f}%')
    plt.plot(error_percentages)
    plt.xlabel('Iteration') 
    plt.ylabel('Error Percentage')
    plt.title('Error Percentage Over Iterations') 
    plt.show()
   
    
def launch_network_test(label_filename, image_filename, examples=10000):
    input_size = 28 * 28
    hidden_size = 100
    output_size = 10
    epsilon = 0.01  # Learning rate
    
    weights_input_hidden = set_weights(input_size, hidden_size)
    weights_hidden_output  = set_weights(hidden_size, output_size)


    error_percentages = []  # For plotting purpose
    count = 0  # Counter for processed examples

    while count < examples:
        labels = read_labels(label_filename)
        images = read_images(image_filename)
        label, image_index = get_random_pattern(images, labels)

        x_j = propagate_on_retina(images[image_index])
        x_h, pot_h = calculate_hidden_layer_output(x_j, weights_input_hidden)
        x_i, pot_i = calculate_output_layer_output(x_h, weights_hidden_output)

        delta_i = calculate_delta_output_layer(pot_i, label, x_i)
        delta_h = calculate_delta_hidden_layer(pot_h, delta_i, weights_hidden_output, x_h)

        for k in range(output_size):
            weights_hidden_output[k] = learn(weights_hidden_output[k], delta_i[k], epsilon, x_h)

        for j in range(hidden_size):
            weights_input_hidden[j] = learn(weights_input_hidden[j], delta_h[j], epsilon, x_j)

        error_percentage = calculate_error_percentage(delta_i, 100)
        error_percentages.append(error_percentage)

        count += 1 

        if count % 100 == 0:  # Log the progress
            print(f'Processed example {count}/{examples}, Current error percentage: {error_percentage:.2f}%')

    plt.plot(error_percentages)
    plt.xlabel('Example')
    plt.ylabel('Error Percentage')
    plt.title('Error Percentage Over Examples')
    plt.show()

 
def main():
    # # Train base
    launch_network('train-labels.idx1-ubyte', 'train-images.idx3-ubyte')
    # # Test base
    # launch_network('t10k-labels.idx1-ubyte', 't10k-images.idx3-ubyte')
    launch_network_test('t10k-labels.idx1-ubyte', 't10k-images.idx3-ubyte')

if __name__ == "__main__":
    main()