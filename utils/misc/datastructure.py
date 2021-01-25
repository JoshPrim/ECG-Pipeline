import numpy as np


def perform_shape_switch(input):
    input = np.asarray(input)
    length = len(input)
    width = len(input[0])

    output = np.zeros((width, length))

    for row in range(width):
        for item in range(length):
            output[row][item] = input[item][row]

    return output