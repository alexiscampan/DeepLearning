"""Synapse
    Synapse contenant le poids synaptique et le neurone entrant et le sortant
"""


class Synapse:
    def __init__(self, weight):
        self.weight = weight

    def set_weight(self, val):
        self.weight = val

    def set_neuron_in(self, n_in):
        self.n_in = n_in

    def set_neuron_out(self, n_out):
        self.n_out = n_out

    def get_neuron_in(self):
        return self.n_in

    def get_reuron_out(self):
        return self.n_out

    def get_weight(self):
        return self.weight
