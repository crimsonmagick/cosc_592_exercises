from numpy import asarray


class Perceptron:
  def __init__(self, weights, bias):
      self.weights = asarray(weights)
      self.bias = bias


