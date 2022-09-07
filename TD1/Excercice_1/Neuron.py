from typing import List

from Abstract_Neuron import Abs_Neuron
from Abstract_Synapse import Abs_Synapse


def sum_network(l):
    result = 0
    for val in l:
        result += val


def relu(val):
    if val > 0:
        return 1
    else:
        return 0


"""
Neuron 
"""


class Neuron(Abs_Neuron):
    def __init__(
        self, value, layer_n: int, node_n: int, fn_aggr=sum_network, fn_act=relu
    ) -> None:  # TODO
        super()
        self.value = value
        self.layer_n = layer_n
        self.node_n = node_n
        self.fn_aggr = fn_aggr
        self.fn_act = fn_act

    def __str__(self):
        return f"Neuron{self.layer_n}{self.node_n}({self.value})"

    def get_value(self):
        return self.value

    # TODO is typing correct ?
    def add_synapses_in(self, s_in: List[Abs_Synapse]) -> None:
        self.synapses_in = s_in

    def add_synapses_out(self, s_out: List[Abs_Synapse]) -> None:
        self.synapses_out = s_out

    def aggregate(self):
        self.res = []
        for synapse in self.synapses_in:
            n = synapse.get_neuron_in()
            pre_activation = n.get_value() * synapse.get_weight()
            self.res.append(pre_activation)

    def activate(self):
        self.aggregate()
        aggregation = self.fn_aggr(self.res)
        self.fn_act(aggregation)
        self.value = aggregation  # Update the value
