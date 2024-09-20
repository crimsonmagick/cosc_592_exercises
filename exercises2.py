import math

import numpy as np
from math import exp


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x) -> float:
    return x * (1 - x)


if __name__ == '__main__':
    training_data = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])

    XOR_labels = np.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ])

    input_neurons = 2
    hidden_neurons = 3
    output_neurons = 1
    learning_rate = 0.3

    w_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    b_hidden = np.random.uniform(size=(1, hidden_neurons))

    w_output = np.random.uniform(size=(hidden_neurons, output_neurons))
    b_output = np.random.uniform(size=(1, output_neurons))

    for epoch in range(50):
        # print("w_hidden")
        # print(f"{w_hidden}")
        # print("b_hidden")
        # print(f"{b_hidden}")
        # print("w_output")
        # print(f"{w_output}")
        # print("b_output")
        # print(f"{b_output}")
        inp = (np.dot(training_data, w_hidden) + b_hidden)
        input_2_hidden_out = sigmoid(inp)
        z = np.dot(input_2_hidden_out, w_output) + b_output
        output = sigmoid(z)
        print("output")
        print(f"{output}")
        # derivative of mean squared error (MSE) - we're trying to optimize for MSE
        error = output - XOR_labels
        avg_error = np.average(np.abs(error))
        print(f"error: {avg_error}")
        if avg_error < 0.01:
            break
        g_out = error * sigmoid_prime(output)
        w_output -= learning_rate * np.dot(input_2_hidden_out.T, g_out)
        b_output -= learning_rate * np.sum(g_out, axis=0, keepdims=True)

        g_hidden = np.dot(g_out, w_output.T) * sigmoid_prime(input_2_hidden_out)
        w_hidden -= learning_rate * np.dot(training_data.T, g_hidden)
        b_hidden -= learning_rate * np.sum(g_hidden, axis=0)

