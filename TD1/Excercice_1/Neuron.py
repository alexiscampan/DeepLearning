
import Synapse

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
Synapse contenant le poids synaptique et le neurone entrant et le sortant
"""


class Neuron:
    def __init__(self, value, fn_aggr=sum_network, fn_act=relu) -> None:  # TODO
        self.value = value
        self.fn_aggr = fn_aggr
        self.fn_act = fn_act

    def add_synapses_in(self, s_in) -> None:
        self.synapses_in = s_in
    
    def add_synapses_out(self, s_out) -> None : 
        self.synapses_out = s_out

    def aggregate(self):
        self.res = []
        for synapse in self.synapses_in:
            n = synapse.get_neuron()
            pre_activation = n.get_value() * synapse.get_weight()
            self.resres.append(pre_activation)

    def activate(self):
        self.aggregate()
        aggregation = self.fn_aggr(self.res)
        self.fn_act(aggregation)
        self.value = aggregation # Update the value

