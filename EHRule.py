"""Train RNN."""

# Author: Amina Keldibek

import numpy as np
from FeedbackNet import FeedbackNet


class EHRule:
    """sdfsdf."""

    def __init__(self, reservoir, learn_rate_init, t_avg, decay_const):
		"""
		Parameters
		----------

		reservoir : ReservoirNodes
			Reservoir node with leaky integrator neurons.

		learn_rate_init : float
			Initial learning rate.

		t_avg : float
			Average time for decaying learning rate.

		decay_const : int
			Decay constant.
        """
		self.__res = reservoir
		self.__t_avg = t_avg
		self.learn_rate_init = learn_rate_init
		self.decay_const = decay_const

    def train(self, output, step, inp=None):
		"""Train RNN.
		Parameters:
		"""
		alpha = 1 - self.__res.time_step / self.__t_avg
		out_size = output.shape[1]
		weights_out = np.zeros((self.__res.network_size, out_size))
		z = np.zeros((1, out_size))
		plp = 0
		zlp = np.zeros((1, out_size))
		out = np.zeros((output.size, out_size))

		for idx in xrange(0, output.shape[0], step):
			r = self.__res.simulate(z, inp)
			z = np.dot(r, weights_out) + np.random.rand(1, out_size) - 0.5

			p = -np.sum(np.power(z - output[idx], 2))
			plp = alpha * plp + (1 - alpha) * p
			zlp = alpha * zlp + (1 - alpha) * z

			if p > plp:
				eta = self.learn_rate_init / (1 + idx * self.__res.time_step / self.decay_const)
				dw = eta * np.multiply(np.transpose(z-zlp), r)
				weights_out = weights_out + np.transpose(dw)
			out[idx, :] = zlp

		model = FeedbackNet(self.__res, weights_out, 0.5)
		return model, out
