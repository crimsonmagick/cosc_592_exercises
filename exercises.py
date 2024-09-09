import math, numpy
from math import exp

from numpy import asarray, dot, zeros


def relu(x: float):
    if x < 0:
        return 0
    return x


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def step_function(x: float) -> float:
    return 1 if x >= 0 else 0


def initialize_perceptron(weights, bias, activation):
    def process(inputs):
        return activation(dot(weights, inputs) + bias), weights, bias
    return process

def train_perceptron(initial_perceptron, epochs, cases):
    r = 0.1
    inputs = []
    # for case in cases:
    #     inputs.append(arr[:-1])





if __name__ == '__main__':
    init_weights = zeros(2)
    init_bias = 0
    forward_pass = initialize_perceptron(init_weights, init_bias, step_function)
    result, weights, bias = forward_pass(asarray([0, 1]))
    print(f"result={result}, weights={weights}, bias={bias}")
    cases = [[0,0,0],[0,1,1],[1,0,1],[1,1,1]]
