from typing import List
import numpy as np
import pandas as pd
import random
from Abstract_Neuron import Abs_Neuron
from Abstract_Synapse import Abs_Synapse
from Neuron import Neuron, heavy
from Synapse import Synapse
from sklearn.metrics import accuracy_score


class MutiLayerPerceptron:
    """ 
    This object instantiates an Multi Layer Perceptron implementation. 

    A Multi Layer Perceptron is a fully connected, feed forward neural network.
    """

    def __init__(self, X: pd.DataFrame, layers: List) -> None:
        """
        A MLP accepts multiple layers. 
        The first layer has X.shape[1] neurons. 

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
        """
        Create a List (layer) of Lists(neurons) that holds all the neurons. 
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
        """
        Create a List (layer i to layer i+1) of Lists (synapses) that holds all the synapses. 
        """
        self.SYNAPSES: List[List[Abs_Synapse]] = []
        for index, out_layer in enumerate(self.NEURONS[slice(1, len(self.NEURONS))]):
            Synapses_li_to_li_plus_1 = []
            layer_minus_1 = self.NEURONS[index]
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
        """
        Proves connections
        """
        for index, layer in enumerate(self.SYNAPSES):
            print("\n")
            print(f"Connections layer {index} to {index + 1}")
            for synapse in layer:
                print(synapse)

    def prove_neurons(self) -> None:
        """
        Proves connections
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
                      " is in for  " + synapse.__str__())
                for neuron in layer
                for synapse in neuron.s_in
            ]

            [
                print("Node " + synapse.get_neuron_out().__str__() +
                      " is out for " + synapse.__str__())
                for neuron in layer
                for synapse in neuron.s_out
            ]

    def forward(self):
        """
        Makes a single forward pass

        Returns:
            float: prediction
        """
        for layer in self.NEURONS[1:]:
            for node in layer:
                node.forward()

        return self.NEURONS[-1][0].value

    def backward(self, update):
        """
        Updates the weights of all the synapses

        Args:
            update (float): c-o in hebb' law
        """
        for layer in self.SYNAPSES:
            for synapse in layer:
                synapse.backward(update)

    def train(self, X, y, epochs=100, verbose: int = 2):
        """ Train the NN a number of times.
        Each train epoch is a forward and backward pass of all train instances.

        Args:
            epochs (int, optional): nb of epochs. 
            X (pd.DataFrame): Training data.
            y (pd.DataFrame): Training targets. 

        Returns:
            List[List[float]]: List of the losses per epoch
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
                if verbose > 1:
                    print(
                        f"Epoch {epoch}. line : {line[0]} Pred : {pred}  Target : {y[line[0]]}")
                if to_update != 0:
                    self.backward(to_update)
                loss.append(to_update)

            # print mean abs loss
            abs_loss = np.array(np.abs(loss)).mean()
            y_pred = self.predict(X)
            acc = accuracy_score(y, y_pred)
            if verbose > 0:
                print(
                    f"Epoch {epoch}. Absolute Loss {abs_loss}. Accuracy {acc}")
            if verbose > 1:
                print("\n")
            if abs_loss == 0:  # kind of an early stop.
                break
            LOSS.append(loss)
        return LOSS

    def predict(self, X_test: pd.DataFrame):
        """
        Makes a forward pass given the X_test inputs. 

        Args:
            X_test (pd.DataFrame): usually test data

        Returns:
            List[float]: prediction for every i row in X_test
        """
        preds = []
        for line in X_test.iterrows():
            # init first layer:
            for neuron_index, neuron in enumerate(self.NEURONS[0]):
                neuron.value = line[1][neuron_index]
            preds.append(self.forward())
        return preds
