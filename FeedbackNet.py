"""RNN model."""

import numpy as np

class FeedbackNet:
"""Create RNN model."""

    def __init__(self, reservoir, weights, noise_level):
        """
        Parameters
        ----------
        reservoir : ReservoirNodes
            Reservoir node with leaky integrator neurons.

        weights : numpy.ndarray, shape = [number of neurons, number of readout neurons]
            Output synapes weights

        noise_level : float
            Noise in the firing rate of readout neurons, from a uniform distribution in the interval [-0.5 0.5]
        """        
        self.res = reservoir
        self.weights = weights
        self.noise_level = 2 * noise_level
        self.feedback = 0.5 * np.random.randn(1, self.weights.shape[1])
        self.__outsize = self.weights.shape[1]

    def simulate(self, inp=None):
        out = np.zeros((inp.shape[0], self.__outsize))
        idx = 0

        for i in inp:
            r = self.res.simulate(i, self.feedback)
            self.feedback = np.dot(r, self.weights) + self.noise_level*(np.random.rand(1, self.__outsize)-0.5)
            out[idx, :] = np.transpose(self.feedback)
            idx = idx+1
        self.feedback = self.feedback

        return out

    def simulateNoInput(self, steps):
        """Generate RNN model output.
        Parameters
        ----------
        steps : int
            Number of samples to generate.

        Returns
        -------
        out : numpy.ndarray shape = [number of samples, number of readout neurons]
            RNN output.
        """
        out = np.zeros((steps, self.__outsize))

        for k in xrange(0, steps):
            r = self.res.simulate(self.feedback)
            self.feedback = np.dot(r, self.weights) + self.noise_level*(np.random.rand(1, self.__outsize)-0.5)
            out[k, :] = self.feedback

        return out
