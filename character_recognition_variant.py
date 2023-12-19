import random

from helper import heaviside, get_file_content, get_file_output_value
from character_recognition import normalize_weights, update_weight

def recognize_character_variant(filename, num_weights, epsilon):
    
    W = initialize_weights_variant(num_weights)
    normalized_weights = normalize_weights(W)
    
    file_content = get_file_content(filename)
    file_output_value = get_file_output_value(file_content)
    X = []
    propagate_on_retina_variant(file_content, X)
    output = calculate_output_neuron_variant(X, normalized_weights)
    error = file_output_value - output
    update_weight(normalized_weights, epsilon, error, X)
    output = calculate_output_neuron_variant(X, normalized_weights)

    return output, error

def propagate_on_retina_variant(file_content, X):
    for line in file_content[:-1]:  # ignore last line
        for char in line:
            if char == "*":
                X.append(1)
            elif char == ".":
                X.append(0)
            else:
                raise ValueError(f"Invalid character: {char}")
    X.append(1) # add X_j+1 = 1
    
def initialize_weights_variant(num_weights):
    return [random.uniform(-1, 1) for _ in range(num_weights + 1)]

def calculate_output_neuron_variant(X, W):
    potential = calculate_output_neuron_potential_variant(X, W)
    return heaviside(potential)

def calculate_output_neuron_potential_variant(X, W):
    return sum(x * w for x, w in zip(X, W))