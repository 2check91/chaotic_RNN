"""Reservoir node with leaky integrator neurons."""

# Author: Amina Keldibek

from math import sqrt
from scipy import sparse
from numpy import random, tanh, dot


class ReservoirNodes:
    """Create reservoir node."""

    def __init__(self, input_size, out_size, network_size, time_const,
                 connect_prob, chaoticity_level, noise_level, time_step):
        """
        Parameters
        ----------
        input_size : integer
            Size of input signal to the network.

        out_size : integer
            Number of readout neurons.

        network_size : integer
            Number of neurons.

        time_const : float
            Membrane time constant.

        connect_prob : float
            Probability of neurons connectivity in RNN.

        chaoticity_level : float
            Parameter for different dynamic regime of the network
            (from ordered to chaotic, good values are 1.5-1.7).

        noise_level : float
            Exploration noise.

        time_step : float
            Simulation time step.

        """
        self.input_size = input_size
        self.network_size = network_size
        self.__time_const = time_const
        self.noise_level = 2 * noise_level
        self.time_step = time_step

        self.__R = sparse.rand(network_size, network_size,
                               density=connect_prob, format='csr')
        scale = 1 / sqrt(connect_prob * network_size)
        self.__R = self.__R * scale * chaoticity_level * time_step

        self.__weights_feedback = (random.rand(out_size, network_size) - 0.5)
        self.__weights_feedback = 2 * time_step * self.__weights_feedback

        self.__weights_input = (random.rand(input_size, network_size) - 0.5)
        self.__weights_input = 2 * time_step * self.__weights_input

        self.__states = 0.5 * random.randn(1, network_size)

        self.__prev_out = tanh(self.__states)

    def simulate(self, z, inp=None):
        """Simulate RNN dynamics.
        Parameters
        ----------
        z : numpy.ndarray
            Readout output.

        inp : numpy.ndarray
            Control input to the network.

        Returns
        -------
        __prev_out : numpy.ndarray
            Previous output.

        """
        if inp is None:
            self.__states = ((1 - self.time_step / self.__time_const) * \
                             self.__states + \
                             (self.__prev_out * self.__R + \
                              dot(z, self.__weights_feedback)) / \
                             self.__time_const)
        else:
            self.__states = ((1 - self.time_step / self.__time_const) * \
                             self.__states + \
                             (self.__prev_out*self.__R  + \
                              dot(z, self.__weights_feedback) + \
                              dot(inp, self.__weights_input)) / \
                             self.__time_const)

        self.__prev_out = tanh(self.__states) + self.noise_level * \
                              (random.rand(1, self.network_size) - 0.5)

        return self.__prev_out
