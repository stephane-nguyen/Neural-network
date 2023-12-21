import os
import struct
import random
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

def display_random_pattern(images, labels):
    index = random.randint(0, len(images) - 1)
    plt.imshow(images[index], cmap='gray')
    plt.title(f'Label: {labels[index]}')
    plt.show()
    print(f'Ydk Class: {labels[index]}')
    return labels[index], index

def set_weights(input_size):
    weights = initialize_weights(input_size)
    weights = normalize_weights(weights)
    return weights

def initialize_weights(num_weights):
    return [random.uniform(-1, 1) for _ in range(num_weights)]

def normalize_weights(W):
    total = sum(W)
    return [w / total for w in W]

def propagate_on_retina(X):
    return X / 255

def calculate_output_neuron(potential):
    return sigmoid(potential)


def calculate_sum(X, W):
    return sum(x * w for x, w in zip(X, W))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_delta_output_layer(pot_i, labels, x_i):
    delta = []
    for i in range(len(x_i)):
        delta.append(sigmoid_derivative(pot_i) * labels[i] - x_i[i])
    return delta

def calculate_delta_hidden_layer(delta_i, pot_h, weights, x_h):
    delta = []
    for _ in range(len(x_h)):
        delta.append(sigmoid_derivative(pot_h) * calculate_sum(delta_i, weights))
    return delta

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def learn(weights, delta_i, epsilon, x_j):
    for i in range(len(weights)):
        weights[i] += epsilon * delta_i * x_j
    return weights

def calculate_error_percentage(delta_i, p):
    print(delta_i)
 
    all_deltas = np.concatenate(delta_i)
    # Calculate the sum of the absolute values of the errors
    error_sum = np.sum(abs(all_deltas))
    # Return the mean error percentage
    print(error_sum)
    return error_sum / (p * len(delta_i))

def main():
    input_size = 28 * 28
    hidden_size = 100
    output_size = 10
    threshold = 0.04
    error_percentage = float("inf")
    while error_percentage > threshold:
      
        weights = set_weights(input_size)
        weights_input_hidden = set_weights(hidden_size)

        labels = read_labels('train-labels.idx1-ubyte')
        images = read_images('train-images.idx3-ubyte')
        pattern, p = display_random_pattern(images, labels)

        x_j = [propagate_on_retina(image) for image in images]

        pot_h = calculate_sum(x_j, weights)
        x_h = calculate_output_neuron(pot_h)

        pot_i = calculate_sum(x_h, weights_input_hidden)
        x_i = calculate_output_neuron(pot_i)

        delta_i = calculate_delta_output_layer(pot_i, labels, x_i)
        delta_h = calculate_delta_hidden_layer(delta_i, x_i, weights_input_hidden, x_h)
        error_percentage = calculate_error_percentage(delta_i, p)

if __name__ == "__main__":
    main()