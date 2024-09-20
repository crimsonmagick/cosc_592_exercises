import math

import numpy as np
from math import exp


def relu(x: float):
    if x < 0:
        return 0
    return x


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def step_function(x: float) -> float:
    return 1 if x >= 0 else 0


def initialize_perceptron(weights, bias, activation):
    def forward_pass(inputs):
        return activation(np.dot(weights, inputs) + bias), weights, bias

    return forward_pass


def train_perceptron(activation, cases, epochs, learning_rate):
    perceptron = initialize_perceptron(np.zeros(2), 0, activation)
    case_arrays = [np.array(case) for case in cases]
    for epoch in range(epochs):
        for case in case_arrays:
            inputs = case[:-1]
            expected = case[-1]
            actual, weights, bias = perceptron(inputs)
            learning_coefficient = learning_rate * (expected - actual)
            adjusted_weights = np.add(weights, np.multiply(inputs, learning_coefficient))
            adjusted_bias = bias + learning_coefficient
            print(f"epoch={epoch}, inputs={inputs}, actual={actual}, expected={expected}, weights={weights}, "
                  f"bias={bias}, learning_coefficient={learning_coefficient}, "
                  f"adjusted_weights={adjusted_weights}, adjusted_bias={adjusted_bias}")
            perceptron = initialize_perceptron(adjusted_weights, adjusted_bias, activation)
    return perceptron


def and_truth_table():
    return [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 1],
    ]


def or_truth_table():
    return [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]


def xor_truth_table():
    return [
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]


def train():
    or_perceptron = train_perceptron(step_function, or_truth_table(), 10, 0.3)
    and_perceptron = train_perceptron(relu, and_truth_table(), 10, 0.1)
    xor_perceptron = train_perceptron(step_function, xor_truth_table(), 10, 0.1)

    # # f(x) = x^2 - 4*x
    # # 2xcosx - x^2sinx-1/10


def e(x):
    return (2 * x * math.cos(x) - x * 2 * math.sin(x) - 1) / 10
def e(x):
    return 2*x - 4


if __name__ == '__main__':
    x = 11
    r = .1
    for _ in range(10):
        x = x - r * e(x)
    print(x)
