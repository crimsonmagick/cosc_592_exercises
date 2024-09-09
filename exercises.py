import math
from math import exp


def relu(x: float):
    if x < 0:
        return 0
    return x


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def sigmoid2(x: float) -> float:
    return 1 / (1 + math.e ** -x)


def neuron(weights, bias, inputs) -> float:
    accumulation = 0
    for index in range(len(weights)):
        accumulation += weights[index] * inputs[index]
    return sigmoid(accumulation + bias)


if __name__ == '__main__':
    print(neuron([0.97, 0.12], 2, [0.5, 0.2]))
