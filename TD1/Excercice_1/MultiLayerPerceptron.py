# %%
from typing import List

import pandas as pd

from Abstract_Neuron import Abs_Neuron
from Abstract_Synapse import Abs_Synapse
from Neuron import Neuron
from Synapse import Synapse

#%%
toy = [[1, 1, 1], [0, 0, 0]]
y = [1, 1]
X = pd.DataFrame(toy)
# %%
class MutiLayerPerceptron:
    def __init__(self, X, layers: List) -> None:
        self.n0_nodes = X.shape[1]  # columns are also nodes
        self.layers = layers.copy()
        self.layers.insert(0, self.n0_nodes)
        self.create_neurons()
        self.create_synapses()
        print(f"# SUMMARY : MultiLayerPercetron with architecture : {self.layers}")

    def create_neurons(self):
        self.NEURONS: List[List[Abs_Neuron]] = []
        for index, n_neurons in enumerate(self.layers):
            assert n_neurons > 0
            neurons_in: List[Abs_Neuron] = []
            for i in range(n_neurons):
                rnd = 1
                neurons_in.append(Neuron(rnd, index, i))

            self.NEURONS.append(neurons_in)

    def create_synapses(self):
        """
        for each output node, connect all input nodes
        """
        self.SYNAPSES: List[List[Abs_Synapse]] = []
        for index, out_layer in enumerate(self.NEURONS[slice(1, len(self.NEURONS))]):
            Synapses_li_to_li_plus_1 = []
            layer_minus_1 = self.NEURONS[index]
            print(layer_minus_1)
            # out layer [Neuron1, Neuron2, Neuron3]
            for out_node in out_layer:
                # Neuron1 => connect to all out_layer-1
                for in_node in layer_minus_1:
                    # create the synapse
                    w = 0.5
                    synapse = Synapse(w)

                    # Add neuron reference to synapse
                    synapse.set_neuron_in(in_node)
                    synapse.set_neuron_out(out_node)

                    # Add synapse references to each node
                    in_node.add_synapse_in(synapse)
                    out_node.add_synapse_out(synapse)

                    # save
                    Synapses_li_to_li_plus_1.append(synapse)

            self.SYNAPSES.append(Synapses_li_to_li_plus_1)

    def prove_synapses(self):
        for index, layer in enumerate(self.SYNAPSES):
            print("\n")
            print(f"Connections layer {index} to {index + 1}")
            for synapse in layer:
                print(synapse)

    def prove_neurons(self) -> None:

        print(
            "We will see that none of the Nodes in Layer 0 is the output node of a synapse (No previous synapses)"
        )

        print("\n")

        print(
            "We will see that none of the Nodes in the last layer is the input to further synapses (No further synapses)"
        )

        print("\n")

        for index, layer in enumerate(self.NEURONS):
            print("\n")
            print(f"Connected Neurons layer {index}")
            [
                print("Node is in for  " + synapse.__str__())
                for neuron in layer
                for synapse in neuron.s_in
            ]

            [
                print("Node is out for " + synapse.__str__())
                for neuron in layer
                for synapse in neuron.s_out
            ]


# %%

mlp = MutiLayerPerceptron(X, [2, 2, 1])
mlp.prove_synapses()

#%%
mlp.prove_neurons()

#%%
# train this thing
mlp.NEURONS[0][0].activate()
