import random

def get_file_content(file_to_read):
    with open(file_to_read, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def choose_random_file(files):
    return random.choice(files)

def get_file_output_value(file_content):
    return int(file_content[-1])

def heaviside(x):
    return 1 if x >= 0 else 0
