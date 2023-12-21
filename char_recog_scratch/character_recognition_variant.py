import random

from helper import heaviside, get_file_content, get_file_output_value
from character_recognition import update_weight

def recognize_character_variant(filename, weights, epsilon):
    file_content = get_file_content(filename)
    file_output_value = get_file_output_value(file_content)
    
    X = []
    propagate_on_retina_variant(file_content, X)
    output = calculate_x_i_variant(X, weights)
    error = file_output_value - output

    if error != 0:
        update_weight(weights, epsilon, error, X)

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

def calculate_x_i_variant(X, W):
    potential = calculate_pot_i_variant(X, W)
    return heaviside(potential)

def calculate_pot_i_variant(X, W):
    pot_i = sum(W[i] * X[i] for i in range(len(X)))
    return pot_i