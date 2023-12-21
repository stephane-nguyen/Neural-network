import random

from helper import get_file_content, get_file_output_value, heaviside

def recognize_character(filename, weights, theta, epsilon):
    file_content = get_file_content(filename)
    file_output_value = get_file_output_value(file_content)
    
    X = []
    propagate_on_retina(file_content, X)
    
    output = calculate_output_neuron(X, weights, theta)
    error = file_output_value - output

    if error != 0:
        update_weight(weights, epsilon, error, X)

    return output, error


def initialize_weights(num_weights):
    return [random.uniform(-1, 1) for _ in range(num_weights)]

def normalize_weights(W):
    total = sum(W)
    return [w / total for w in W]

def propagate_on_retina(file_content, X):
    for line in file_content[:-1]:  # ignore last line
        for char in line:
            if char == "*":
                X.append(1)
            elif char == ".":
                X.append(0)
            else:
                raise ValueError(f"Invalid character: {char}")

def calculate_output_neuron(X, W, theta):
    potential = calculate_output_neuron_potential(X, W, theta)
    return heaviside(potential)

def calculate_output_neuron_potential(X, W, theta):
    return sum(x * w for x, w in zip(X, W)) - theta

def update_weight(normalized_weights, epsilon, error, X):
    for i in range(len(normalized_weights)):
        normalized_weights[i] += epsilon * error * X[i]