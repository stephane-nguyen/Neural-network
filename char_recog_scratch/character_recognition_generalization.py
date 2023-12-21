import random 
import matplotlib.pyplot as plt

from helper import heaviside, get_file_content, get_file_output_value
from character_recognition import *

def plot_generalization_curve(file_name, num_weights, theta):
    noise_levels, error_rates = build_generalization_curve(file_name, num_weights, theta)
    plt.plot(noise_levels, error_rates, label=f"Pattern from {file_name}")

def build_generalization_curve(file_name, num_weights, theta, num_trials=50):
    noise_levels = range(0, 101, 10)  # Noise levels from 0% to 100% in steps of 10%
    error_rates = []

    for noise in noise_levels:
        errors = count_errors_for_noise_level(file_name, num_weights, theta, noise, num_trials)
        error_rate = errors / num_trials
        error_rates.append(error_rate)

    print(error_rates)
    return noise_levels, error_rates

def count_errors_for_noise_level(filename, num_weights, theta, noise_percentage, num_patterns=50):
    errors = 0
    W = initialize_weights(num_weights)
    normalized_weights = normalize_weights(W)

    for _ in range(num_patterns):
        output, error = recognize_character_with_noise(filename, normalized_weights, theta, noise_percentage)
        if error != 0:
            errors += 1

    return errors


def recognize_character_with_noise(filename, weights, theta, noise_percentage):
    file_content = get_file_content(filename)
    file_output_value = get_file_output_value(file_content)
    
    X = []
    propagate_on_retina(file_content, X)
    inverse_input_pixels(X, noise_percentage)  # Add noise

    output = calculate_x_i(X, weights, theta)
    # error = file_output_value - output
    potential = calculate_pot_i(X, weights, theta)
    error = file_output_value - potential  # Widrow-Hoff error
    return output, error

def inverse_input_pixels(X, noise_percentage):
    num_pixels = len(X)
    num_noisy_pixels = int(num_pixels * noise_percentage / 100) 
    pixels_to_invert = random.sample(range(num_pixels), num_noisy_pixels)

    for i in pixels_to_invert:
        X[i] = 1 - X[i]  
    return X

