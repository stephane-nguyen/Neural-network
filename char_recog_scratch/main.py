from helper import *
from character_recognition import *
from character_recognition_variant import *
from character_recognition_generalization import *

import numpy as np
import matplotlib.pyplot as plt

def show_bar(errors, errors_variant):
    
    fig, ax = plt.subplots()

    # Plotting the errors for each variant
    ax.stairs(errors, label='Errors', linewidth=2.5)
    ax.stairs(errors_variant, label='Errors Variant', linewidth=2.5, linestyle='dashed')

    # Adjusting the x-axis and y-axis limits and ticks
    max_iterations = max(len(errors), len(errors_variant))
    ax.set_xlim(0, max_iterations)
    ax.set_xticks(np.arange(1, max_iterations + 1))

    max_error = max(max(errors), max(errors_variant))
    ax.set_ylim(0, max_error)
    ax.set_yticks(np.arange(0, max_error + 1))

    # Adding labels and legend
    plt.xlabel("Iteration")
    plt.ylabel("Total Error")
    plt.title("Error Comparison")
    plt.legend()
    plt.show()
    
def show_curve(files, num_weights, theta):
    
    # Plotting both curves in the same plot
    plt.figure()
    plot_generalization_curve(files[0], num_weights, theta)  
    plot_generalization_curve(files[1], num_weights, theta)    

    plt.xlabel('Noise Level (%)')
    plt.ylabel('Error Rate')
    plt.title('Generalization Curves')
    plt.legend()
    plt.show()
    
def main():
    num_weights = 48  # 6x8 grid
    theta = 0.5
    epsilon = 0.01
    files = ["zero.txt", "un.txt"]

    W = initialize_weights(num_weights)
    normalized_weights = normalize_weights(W)


    err, err_other_file = float("inf"), float("inf")
    total_err = abs(err) + abs(err_other_file)
    errors = []

    while total_err > 0:
        file_name = choose_random_file(files)
        other_file = get_other_file(file_name)

        output, err = recognize_character(file_name, normalized_weights, theta, epsilon)
        output_other_file, err_other_file = recognize_character(other_file, normalized_weights, theta, epsilon)

        total_err = abs(err) + abs(err_other_file)
        errors.append(total_err)


    W_variant = initialize_weights_variant(num_weights) 
    normalized_weights_variant = normalize_weights(W_variant)

    err_variant, err_other_file_variant = float("inf"), float("inf")
    total_err_variant = abs(err_variant) + abs(err_other_file_variant)
    errors_variant = []

    while total_err_variant > 0:
        file_name = choose_random_file(files)
        other_file = get_other_file(file_name)

        output_variant, err_variant = recognize_character_variant(file_name, normalized_weights_variant, epsilon)
        output_other_file_variant, err_other_file_variant = recognize_character_variant(other_file, normalized_weights_variant, epsilon)

        total_err_variant = abs(err_variant) + abs(err_other_file_variant)
        errors_variant.append(total_err_variant)

    show_bar(errors, errors_variant)
    show_curve(files, num_weights, theta)

if __name__ == "__main__":
    main()