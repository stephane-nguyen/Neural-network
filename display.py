import matplotlib.pyplot as plt
import numpy as np
plt.style.use('_mpl-gallery')

def log(file_name, output, normalized_weights):
    print("Processed file:", file_name)
    print("Network output:", output)
    print("Weights after update:", normalized_weights)

def show_plot_stairs(errors, iteration):
    fig, ax = plt.subplots()

    ax.stairs(errors, linewidth=2.5)

    ax.set(xlim=(0, 1), xticks=np.arange(1, iteration + 1),
        ylim=(0, 1), yticks=np.arange(1, 3))
    plt.xlabel("Total error")

    plt.show()