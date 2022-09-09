from typing import List

from Abstract_Neuron import Abs_Neuron
from Abstract_Synapse import Abs_Synapse


def sum_network(l):
    result = 0
    for val in l:
        result += val
    return result


def relu(val):
    if val > 0:
        return val
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
        self.s_in: List[Abs_Synapse] = []
        self.s_out: List[Abs_Synapse] = []
        self.fn_aggr = fn_aggr
        self.fn_act = fn_act
        self.to_update = True  # on veut pas update la première couche, où to_update sera False

    def __str__(self):
        return f"Neuron{self.layer_n}{self.node_n}({self.value})"

    def get_value(self):
        return self.value

    def add_synapse_in(self, synapse: Abs_Synapse) -> None:
        self.s_in.append(synapse)

    def add_synapse_out(self, synapse: Abs_Synapse) -> None:
        self.s_out.append(synapse)

    def aggregate(self):
        self.res = []
        # When the node is the output node to some connections,
        # calculate each activation value : in_node value * w
        for synapse in self.s_out:
            n = synapse.get_neuron_in()
            pre_activation = n.get_value() * synapse.get_weight()
            self.res.append(pre_activation)

    def forward(self):
        self.aggregate()
        aggregation = self.fn_aggr(self.res)
        print("Aggregation in node: ", self.res)
        self.fn_act(aggregation)
        self.value = aggregation  # Update the value
        return aggregation
