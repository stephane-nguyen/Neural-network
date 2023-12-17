import random

def get_file_content(file_to_read):
    with open(file_to_read, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def choose_random_file(files):
    return random.choice(files)

def get_file_output_value(file_content):
    return int(file_content[-1])

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

def heaviside(x):
    return 1 if x >= 0 else 0

def update_weight(W_normalized, epsilon, error, X):
    for i in range(len(W_normalized)):
        W_normalized[i] += epsilon * error * X[i]

def log(file_name, output, W_normalized):
    print("Processed file:", file_name)
    print("Network output:", output)
    print("Weights after update:", W_normalized)

##################### VARIANT ######################
def propagate_on_retina_variant(file_content, X):
    for line in file_content[:-1]:  # ignore last line
        for char in line:
            if char == "*":
                X.append(1)
            elif char == ".":
                X.append(0)
            else:
                raise ValueError(f"Invalid character: {char}")
    X.append(1)
def initialize_weights_variant(num_weights):
    return [random.uniform(-1, 1) for _ in range(num_weights + 1)]

def calculate_output_neuron_variant(X, W):
    potential = calculate_output_neuron_potential_variant(X, W)
    return heaviside(potential)

def calculate_output_neuron_potential_variant(X, W):
    return sum(x * w for x, w in zip(X, W))

##################### GENERALISATION ######################

def inverse_input_pixels(X, pourcentage):

    num_pixels = len(X)
    num_noisy_pixels = num_pixels * pourcentage / 100

    pixel_positions = random.sample(X, k=num_noisy_pixels)

    for i in pixel_positions:
        if X[i] != 0:
            X[i] = 1 / X[i]
    return X

def get_num_errors():
    pass

def build_generalization_curve_file_zero(file_name="zero.txt"):
    pass

def build_generalization_curve_file_one(file_name="un.txt"):
    pass



def main():
    num_weights = 48  # 6x8 grid
    theta = 0.5
    epsilon = 0.01
    files = ["zero.txt", "un.txt"]

    W = initialize_weights(num_weights)
    W_normalized = normalize_weights(W)
    
    file_name = choose_random_file(files)
    file_content = get_file_content(file_name)
    file_output_value = get_file_output_value(file_content)

    X = []
    propagate_on_retina(file_content, X)

    output = calculate_output_neuron(X, W_normalized, theta)

    error = file_output_value - output
    update_weight(W_normalized, epsilon, error, X)
    output = calculate_output_neuron(X, W_normalized, theta)

    log(file_name, output, W_normalized)


    ############### VARIANT ######################
    X_variant = []
    W_variant = initialize_weights_variant(num_weights)
    W_normalized_variant = normalize_weights(W_variant)
    propagate_on_retina_variant(file_content, X_variant)
    output_variant = calculate_output_neuron_variant(X_variant, W_normalized_variant)
    update_weight(W_normalized_variant, epsilon, error, X_variant)
    output_variant = calculate_output_neuron_variant(X_variant, W_normalized_variant)
    log(file_name, output_variant, W_normalized_variant)

if __name__ == "__main__":
    main()
