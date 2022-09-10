from ast import Call
from typing import List, Callable
from Abstract_Neuron import Abs_Neuron
from Abstract_Synapse import Abs_Synapse


def sum_network(l: List[float]):
    """  From a list of float values, it calculates the sum of all the values.
    In this context, this function does the sum of all wi*xi values of each input neuron.
    Args:
        l (list[float]): A list with float values

    Returns:
        float : the sum over all the values => the pre-actvation value.
    """
    result = 0
    for val in l:
        result += val
    return result


def relu(val):
    """ Rectified Linear Unit

    Args:
        val (float): The pre-activated output value

    Returns:
        float: The activated value
    """
    if val > 0:
        return val
    else:
        return 0


def heavy(val):
    """ Heavy side utility function. 
    Returns 1 if the value is > 0 and 0 otherwise. 

    Args:
        val (float): Tipically the NN output value 

    Returns:
        Int: Int(condition)
    """
    return int(val > 0)


class Neuron(Abs_Neuron):
    """ This is a simple neuron that is aware of its place in the network and the synapses connected to it. 

    Args:
        Abs_Neuron (Abs_Neuron): Abstract class
    """

    def __init__(
        self, value: float, layer_n: int, node_n: int, fn_aggr: Callable = sum_network, fn_act: Callable = relu
    ) -> None:
        """ 

        Args:
            value (float): Initial value
            layer_n (int): _description_
            node_n (int): _description_
            fn_aggr (function, optional): The sum over pre-activated values fn. Defaults to sum_network.
            fn_act (function, optional): The activation fn. Defaults to relu.
        """
        super()
        self.value = value
        self.layer_n = layer_n
        self.node_n = node_n
        self.s_in: List[Abs_Synapse] = []
        self.s_out: List[Abs_Synapse] = []
        self.fn_aggr = fn_aggr
        self.fn_act = fn_act
        self.to_update = True  # on veut pas update la première couche, où to_update sera False

    def __str__(self) -> None:
        """ print method
        """
        return f"Neuron{self.layer_n}{self.node_n}({round(self.value,1)})"

    def get_value(self) -> float:
        """ Getter for the value

        Returns:
            float: Internal value
        """
        return self.value

    def add_synapse_in(self, synapse: Abs_Synapse) -> None:
        """ Adds a Synapse to the list of synapses in.
        This list holds the Synapses for which this neuron is the input. 

        Args:
            synapse (Abs_Synapse): A synapse for which this Neuron is the neuron_in attr. 
        """
        self.s_in.append(synapse)

    def add_synapse_out(self, synapse: Abs_Synapse) -> None:
        """ Adds a Synapse to the list of synapses out
        This list holds the Synapses for which this neuron is the output. 

        Args:
            synapse (Abs_Synapse): A synapse for which this Neuron is the neuron_out attr. 

        """
        self.s_out.append(synapse)

    def aggregate(self) -> None:
        """ Creates a list of weigthed input values
        That is, [w1*x1, w2*x2, ..., wn*xn]
        The xi come from the input neurons of the synapses to which this Neuron is the output neuron.
        The wi values come from the synapses to which this Neuron is the output neuron. 
        """
        self.res = []
        # When the node is the output node to some connections,
        # calculate each activation value : in_node value * w
        for synapse in self.s_out:
            n = synapse.get_neuron_in()
            pre_activation = n.get_value() * synapse.get_weight()
            self.res.append(pre_activation)

    def forward(self) -> float:
        """ Calculates the activated value of the neuron. 
        Does the aggregation (fn_aggr) over wi*xi values (coming from the synapses) and activates this value, using the fn_act
        Updates the inner value of the neuron with the activated result
        Returns:
            float: the activated ouput of the neuron
        """
        self.aggregate()
        aggregation = self.fn_aggr(self.res)
        activated = self.fn_act(aggregation)
        self.value = activated  # Update the value
        return activated
