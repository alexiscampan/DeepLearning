# %%
from typing import List
import numpy as np
import pandas as pd
import random
from Abstract_Neuron import Abs_Neuron
from Abstract_Synapse import Abs_Synapse
from Neuron import Neuron, heavy
from Synapse import Synapse


# %%
# a, b, c
# fn => (a and b) or c
toy = [[1, 1, 1],  # true
       [1, 0, 0],  # false
       [0, 0, 0],  # false
       [1, 0, 1],  # false
       [1, 1, 0],  # true
       [0, 0, 1]  # true
       ]
y = [
    1,
    0,
    0,
    0,
    1,
    1
]
X = pd.DataFrame(toy)
# %%


class MutiLayerPerceptron:
    """_summary_
    """

    def __init__(self, X: pd.DataFrame, layers: List) -> None:
        """_summary_

        Args:
            X (pd.DataFrame): _description_
            layers (List): _description_
        """
        self.n0_nodes = X.shape[1]  # columns are also nodes
        self.layers = layers.copy()
        self.X = X
        self.layers.insert(0, self.n0_nodes)
        self.create_neurons()
        self.create_synapses()
        print(
            f"# SUMMARY : MultiLayerPercetron with architecture : {self.layers}")

    def create_neurons(self):
        """_summary_
        """
        self.NEURONS: List[List[Abs_Neuron]] = []
        for index, n_neurons in enumerate(self.layers):
            assert n_neurons > 0
            neurons_in: List[Abs_Neuron] = []
            for i in range(n_neurons):
                rnd = random.uniform(-1, 1)
                if index == len(self.layers) - 1:  # last layer
                    # the last layer has a heavy(0, 1) activation
                    n = Neuron(rnd, index, i, fn_act=heavy)
                else:
                    n = Neuron(rnd, index, i)  # by default, a relu is used

                neurons_in.append(n)

            self.NEURONS.append(neurons_in)

    def create_synapses(self):
        """_summary_
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
                    w = random.uniform(-1, 1)
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
        """_summary_
        """
        for index, layer in enumerate(self.SYNAPSES):
            print("\n")
            print(f"Connections layer {index} to {index + 1}")
            for synapse in layer:
                print(synapse)

    def prove_neurons(self) -> None:
        """_summary_
        """

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
                print("Node " + synapse.get_neuron_in().__str__() +
                      "is in for  " + synapse.__str__())
                for neuron in layer
                for synapse in neuron.s_in
            ]

            [
                print("Node " + synapse.get_neuron_out().__str__() +
                      "is out for " + synapse.__str__())
                for neuron in layer
                for synapse in neuron.s_out
            ]

    def forward(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        for layer in self.NEURONS[1:]:
            for node in layer:
                node.forward()

        return self.NEURONS[-1][0].value

    def backward(self, update):
        """_summary_

        Args:
            update (_type_): _description_
        """
        for layer in self.SYNAPSES:
            for synapse in layer:
                synapse.backward(update)

    def train(self, epochs=100, X=X, y=y):
        """_summary_

        Args:
            epochs (int, optional): _description_. Defaults to 100.
            X (_type_, optional): _description_. Defaults to X.
            y (_type_, optional): _description_. Defaults to y.

        Returns:
            _type_: _description_
        """
        LOSS = []

        for epoch in range(epochs):
            loss = []
            for line in X.iterrows():
                # init first layer:
                for neuron_index, neuron in enumerate(self.NEURONS[0]):
                    neuron.value = line[1][neuron_index]
                pred = self.forward()
                to_update = y[line[0]] - pred
                print(
                    f"Epoch {epoch}. line : {line[0]} Pred : {pred}  Target : {y[line[0]]}")
                if to_update != 0:
                    self.backward(to_update)
                loss.append(to_update)
            LOSS.append(loss)
        return LOSS


# %%
mlp = MutiLayerPerceptron(X, [3, 2, 1])
mlp.prove_synapses()

# %%
mlp.prove_neurons()

# %%
# train this thing

# %%

l = mlp.train()

# %%
