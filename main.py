from display import *
from helper import *
from character_recognition import *
from character_recognition_variant import *
from display import *


def get_other_file(file_name):
    if file_name == "zero.txt":
        return "un.txt"
    return "zero.txt"


def main():
    num_weights = 48  # 6x8 grid
    theta = 0.5
    epsilon = 0.01

    files = ["zero.txt", "un.txt"]
  
    err, err_other_file = float("inf"), float("inf")
    total_err = abs(err) + abs(err_other_file)

    errors = []


    while total_err > 0:
        file_name = choose_random_file(files)
        other_file = get_other_file(file_name)

        output, err = recognize_character(file_name, num_weights, theta, epsilon)
        output_other_file, err_other_file = recognize_character(other_file, num_weights, theta, epsilon)

        total_err = abs(err) + abs(err_other_file)
        errors.append(total_err)



    # Variant
    err_variant, err_other_file_variant = float("inf"), float("inf")
    total_err_variant = abs(err_variant) + abs(err_other_file_variant)

    errors_variant = []

    while total_err_variant > 0:
        file_name = choose_random_file(files)
        other_file = get_other_file(file_name)

        output_variant, err_variant = recognize_character_variant(file_name, num_weights, epsilon)
        output_other_file_variant, err_other_file_variant = recognize_character_variant(other_file, num_weights, epsilon)

        total_err_variant = abs(err_variant) + abs(err_other_file_variant)
        errors_variant.append(total_err_variant)


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

  



if __name__ == "__main__":
    main()