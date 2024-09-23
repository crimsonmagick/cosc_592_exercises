import numpy as np


class TwoLayerNet:

    def __init__(self, input_neuron_count, hidden_neuron_count, output_neuron_count, learning_rate):
        self.input_neuron_count = input_neuron_count
        self.hidden_neuron_count = hidden_neuron_count
        self.output_neuron_count = output_neuron_count
        self.learning_rate = learning_rate
        
        self.input_2_hidden_out = None
        self.output_1_out = None

        self.w_hidden = np.random.uniform(size=(input_neuron_count, hidden_neuron_count))
        self.b_hidden = np.random.uniform(size=(1, hidden_neuron_count))

        self.w_output = np.random.uniform(size=(hidden_neuron_count, output_neuron_count))
        self.b_output = np.random.uniform(size=(1, output_neuron_count))

    @staticmethod
    def _sigmoid(x) -> float:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_prime(sig_out) -> float:
        return sig_out * (1 - sig_out)

    def __string__(self):
        return (f"w_hidden\n{self.w_hidden}"
                f"\nb_hidden\n{self.b_hidden}"
                f"\nw_output\n{self.w_output}"
                f"b_output{self.b_output}")

    def forward(self, input_neurons):
        inp = (np.dot(input_neurons, self.w_hidden) + self.b_hidden)
        self.input_2_hidden_out = self._sigmoid(inp)
        self.output_1_out = np.dot(self.input_2_hidden_out, self.w_output) + self.b_output
        return self._sigmoid(self.output_1_out)

    def backpropgate(self, input_neurons, labels):
        error = self.output_1_out - labels
        g_out = error * self._sigmoid_prime(self.output_1_out)
        self.w_output -= self.learning_rate * np.dot(self.input_2_hidden_out.T, g_out)
        self.b_output -= self.learning_rate * np.sum(g_out, axis=0, keepdims=True)

        g_hidden = np.dot(g_out, self.w_output.T) * self._sigmoid_prime(self.input_2_hidden_out)
        self.w_hidden -= self.learning_rate * np.dot(input_neurons.T, g_hidden)
        self.b_hidden -= self.learning_rate * np.sum(g_hidden, axis=0)
