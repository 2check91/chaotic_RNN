"""
Recurrent Neural Network (RNN).

This is an implementation of the RNN model described in:
Hoerzer, G., Legenstein, R., & Maass, W. (2012).
Emergence of Complex Computational Structures From Chaotic
Neural Networks Through Reward-Modulated Hebbian Learning.
Cerebral Cortex, 24(3), 677-690.
"""

# Author : Amina Keldibek

from ReservoirNodes import ReservoirNodes
from EHRule import EHRule
from scipy.io import loadmat
from numpy import zeros
import matplotlib.pyplot as plt


def train(data):
    """Train RNN.
    Parameters
    ----------
    data: numpy.ndarray, shape = [number of samples, number of readout neurons]
          Data to train RNN

    Returns
    -------
    model : trained RNN model
    train_out : output of RNN during training phase
    """
    network_size = 1000
    input_size = 0
    connect_prob = 0.1
    chaoticity_level = 1.7
    time_const = 10e-3
    noise_level = 0.05
    time_step = 1e-3

    learn_rate_init = 0.001
    t_avg = 2e-3
    decay_const = 20

    reservoir = ReservoirNodes(input_size, data.shape[1], network_size,
                               time_const, connect_prob, chaoticity_level,
                               noise_level, time_step)
    trainer = EHRule(reservoir, learn_rate_init, t_avg, decay_const)

    model, train_out = trainer.train(data, 1)

    return model, train_out


def test(N, model):
    """Generate RNN output
    Parameters
    ----------
    N: integer
          Number of samples to be generated
    model: FeedbackNet object
          RNN model

    Returns
    -------
    model : trained RNN model
    output : output of RNN model of size N
    """
    out_test = model.simulateNoInput(int(N))

    return out_test


def driver():
    data = loadmat('filtered_data.mat')

    trainTime  = 20000
    testTime = 1000
    numOfJoints = 1

    # train model
    model, out_train = train(data['jointTrajFilt'][0:trainTime, 0:numOfJoints])

    # test model
    model.noise_level = 0
    model.feedback = out_train[-1,:]
    out_test = test(testTime, model)

    """# RMS
    rms = zeros(numOfJoints)
    for i in range(0, numOfJoints):
        rms[i] = sqrt(((out_test[:,i] - data['jointTrajFilt'][trainTime:trainTime+testTime,i] ) ** 2).mean())
    print (rms)"""

    # visualize output
    plt.figure(1)
    # RNN output during training phase
    plt.plot(out_train, 'r', data['jointTrajFilt'][0:trainTime-1,0], 'g')
    plt.figure(2)
    # RNN output
    plt.plot(out_test, 'r', data['jointTrajFilt'][trainTime:trainTime+testTime,0], 'g')
    plt.show()


if __name__ == "__main__":
    driver()
