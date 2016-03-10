import numpy as np

class FeedbackNet:

    def __init__(self, reservoir, weights, noise_level):
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
        out = np.zeros((steps, self.__outsize))

        for k in xrange(0, steps):
            r = self.res.simulate(self.feedback)
            self.feedback = np.dot(r, self.weights) + self.noise_level*(np.random.rand(1, self.__outsize)-0.5)
            out[k, :] = self.feedback

        return out
