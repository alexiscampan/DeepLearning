from Abstract_Neuron import Abs_Neuron

"""
    Synapse contenant le poids synaptique et le neurone entrant et le sortant
"""


class Synapse:
    def __init__(self, weight: float):
        self.weight = weight
        self.n_in = None
        self.n_out = None

    def __str__(self):
        if self.n_in and self.n_out:
            return f"Synapse from {self.n_in.__str__()} to {self.n_out.__str__()}"

    def set_weight(self, val: float):
        self.weight = val

    def set_neuron_in(self, n_in: Abs_Neuron):
        self.n_in = n_in

    def set_neuron_out(self, n_out: Abs_Neuron):
        self.n_out = n_out

    def get_neuron_in(self):
        return self.n_in

    def get_neuron_out(self):
        return self.n_out

    def get_weight(self):
        return self.weight

    def backward(self, c):
        lr = 0.001
        xi = self.n_in.value
        difference = c - self.n_out.value
        print(difference)
        if (difference != 0):
            print(f"backward difference {difference}")
            self.weight = self.weight + lr * difference * xi
