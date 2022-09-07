# %%
from typing import List

from Abstract_Neuron import Abs_Neuron
from Abstract_Synapse import Abs_Synapse
from Neuron import Neuron
from Synapse import Synapse


# %%
class MutiLayerPerceptron:
    pass


ex = [2, 2, 2]

# %%
NEURONS: List[List[Abs_Neuron]] = []

for index, n_neurons in enumerate(ex):
    assert n_neurons > 0
    neurons_in: List[Abs_Neuron] = []
    for i in range(n_neurons):
        rnd = 1
        neurons_in.append(Neuron(rnd, index, i))

    NEURONS.append(neurons_in)

# %%
"""
for each output node, connect all input nodes
"""
SYNAPSES: List[List[Abs_Synapse]] = []
for index, out_layer in enumerate(NEURONS[slice(1, len(NEURONS))]):
    Synapses_li_to_li_plus_1 = []
    layer_minus_1 = NEURONS[index]
    print(layer_minus_1)
    # out layer [Neuron1, Neuron2, Neuron3]
    for out_node in out_layer:
        # Neuron1 => connect to all out_layer-1
        for in_node in layer_minus_1:
            w = 0.5
            synapse = Synapse(w)
            synapse.set_neuron_in(in_node)
            synapse.set_neuron_out(out_node)
            Synapses_li_to_li_plus_1.append(synapse)
    SYNAPSES.append(Synapses_li_to_li_plus_1)

# %%
print("EXEMPLE LAYER 0 TO 1")
print(SYNAPSES[0][0])
print(SYNAPSES[0][1])
print(SYNAPSES[0][2])
print(SYNAPSES[0][3])

print("\n")

print("EXEMPLE LAYER 1 TO 2")
print(SYNAPSES[1][0])
print(SYNAPSES[1][1])
print(SYNAPSES[1][2])
print(SYNAPSES[1][3])
# print(SYNAPSES[0][1].get_neuron_in())

#%%
print("NEURONS LAYER 0")
for i in NEURONS[0]:
    print(i)

print("\n")

print("NEURONS LAYER 1")

for i in NEURONS[1]:
    print(i)
# %%
