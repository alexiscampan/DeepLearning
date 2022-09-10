from Abstract_Synapse import Abs_Synapse
from Abstract_Neuron import Abs_Neuron


class Synapse(Abs_Synapse):
    """ The synapse connects to Neurons (in and out) with a weight in the middle. 

    Args:
        Abs_Synapse (Abs_Synapse): Abstract Synapse
    """

    def __init__(self, weight: float) -> None:
        """ Sets the weight

        Args:
            weight (float): The initial weight of the Synapse
        """
        self.weight = weight
        self.n_in = None
        self.n_out = None
        self.lr = 0.1

    def __str__(self) -> None:
        """ Print method

        Returns:
            string: description of the in and out neuron 
        """
        if self.n_in and self.n_out:
            return f"Synapse from {self.n_in.__str__()} to {self.n_out.__str__()}"

    def set_weight(self, val: float) -> None:
        """ Updates the weight

        Args:
            val (float): new weight
        """
        self.weight = val

    def set_neuron_in(self, n_in: Abs_Neuron) -> None:
        """ Places a Neuron as the input neuron of this connection

        Args:
            n_in (Abs_Neuron): The input neuron
        """
        self.n_in = n_in

    def set_neuron_out(self, n_out: Abs_Neuron) -> None:
        """ Places a Neuron as the output neuron of this connection

        Args:
            n_out (Abs_Neuron): The output Neuron
        """
        self.n_out = n_out

    def get_neuron_in(self) -> Abs_Neuron:
        """ Getter for the input neuron

        Returns:
            Abs_Neuron: The current input neuron 
        """
        return self.n_in

    def get_neuron_out(self) -> Abs_Neuron:
        """ Getter for the output neuron

        Returns:
            Abs_Neuron: The current output neuron
        """
        return self.n_out

    def get_weight(self) -> float:
        """ Getter for the weight

        Returns:
            float: The current weight
        """
        return self.weight

    def backward(self, update: float) -> None:
        """ Updates the inner weight given an update value 
        The update is computed as w = w + lr * update * input.value

        Args:
            update (float): The update direction and magnitude
        """
        xi = self.n_in.value
        self.weight = self.weight + self.lr * update * xi
