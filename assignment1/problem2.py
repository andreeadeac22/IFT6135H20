import pickle
import numpy as np
import gzip
import matplotlib.pyplot as plt
from solution import *

class SuperNN(NN):
    def __init__(self,
                 hidden_dims=(784, 256),
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=64,
                 seed=None,
                 activation="relu",
                 data=None
                 ):
        super(SuperNN, self).__init__(hidden_dims, epsilon, lr, batch_size, seed, activation, data)

    def zero_initialization(self,  dims):
        print("Using zero initialization")
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"W{layer_n}"] = np.zeros((all_dims[layer_n-1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def initialize_weights(self, dims):
        print("Using normal initiazation")
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"W{layer_n}"] = np.random.normal(0, 1, (all_dims[layer_n - 1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def plot(self):
        zero_init_train_loss = [2.302421418707069, 2.302274667119889, 2.3021431184461356, 2.3020252225102973,
                                2.3019195827656724, 2.3018249417288428, 2.301740167711852, 2.3016642427439344,
                                2.301596251583872, 2.3015353717321325]
        #normal_init_train_loss = []
        glorot_init_train_loss = [1.8977142252701094, 1.4780891970087835, 1.135595875035603, 0.9079237720563468,
                                  0.7628293519489534, 0.6669359481493732, 0.6003134078832704, 0.5518539542483993,
                                  0.5151949688588041, 0.48646434408492145]
        plt.figure()
        plt.plot(range(10), zero_init_train_loss, 'b', label='Zero initialization')
        #plt.plot(range(10), normal_init_train_loss, 'r', label='Normal initialization')
        plt.plot(range(10), glorot_init_train_loss, 'g', label='Glorot initialization')
        plt.xlabel('epoch')
        plt.ylabel('training loss')
        plt.title('Average training loss using different weight initializations')
        plt.legend()
        plt.savefig('intializations_.jpg')

data = load_mnist()
nn = SuperNN(data=data)
train_logs = nn.train_loop(10)
#nn.plot()