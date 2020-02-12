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
    #def initialize_weights(self, dims):
        print("Using zero initialization")
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"W{layer_n}"] = np.zeros((all_dims[layer_n-1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def normal_initialization(self, dims):
    #def initialize_weights(self, dims):
        print("Using normal initiazation")
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"W{layer_n}"] = np.random.normal(0, 1, (all_dims[layer_n - 1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
            #print("self.weights[fW{layer_n}]" , self.weights[f"W{layer_n}"])
            assert not np.any(np.isnan(self.weights[f"W{layer_n}"]))



    """
    Zero init results
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 0 0.11356 0.1064 2.302421418707069 2.302475402678308
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 1 0.11356 0.1064 2.302274667119889 2.302379743903525
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 2 0.11356 0.1064 2.3021431184461356 2.302296537831699
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 3 0.11356 0.1064 2.3020252225102973 2.302224369853798
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 4 0.11356 0.1064 2.3019195827656724 2.3021619736552696
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 5 0.11356 0.1064 2.3018249417288428 2.3021082167163534
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 6 0.11356 0.1064 2.301740167711852 2.3020620871262993
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 7 0.11356 0.1064 2.3016642427439344 2.3020226816012888
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 8 0.11356 0.1064 2.301596251583872 2.3019891946051874
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 9 0.11356 0.1064 2.3015353717321325 2.301960908480417
    """

    """
    Glorot results
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 0 0.61406 0.6381 1.8601016234556451 1.8422919262847697
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 1 0.74656 0.7737 1.4400731981324928 1.410838792941645
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 2 0.79762 0.8253 1.1124919536438642 1.0739755973707485
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 3 0.82444 0.8456 0.897167160486312 0.8531941920368635
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 4 0.84086 0.8615 0.7593353841920966 0.712886891644657
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 5 0.85256 0.8717 0.6672244247053242 0.6200696906127845
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 6 0.86178 0.8796 0.6021454189180506 0.5553280190452572
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 7 0.86828 0.8876 0.553894626041198 0.507964579021095
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 8 0.87388 0.8918 0.5167791770125256 0.4720431400240831
    Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 9 0.87874 0.8959 0.4873786554075815 0.4439725583926579
    
    """

    def plot(self):
        zero_init_train_loss = [2.302421418707069, 2.302274667119889, 2.3021431184461356, 2.3020252225102973,
                                2.3019195827656724, 2.3018249417288428, 2.30174016771185, 2.3016642427439344,
                                2.301596251583872, 2.3015353717321325]
        normal_init_train_loss = [2.1670133457355707, 1.6282529320101617, 1.412485651398122, 1.1769963806257004,
                                  1.0669246246750133, 0.9999256249361914, 0.9400407181562437, 0.8923009483761754,
                                  0.8431614317669878, 0.8175628259848486]
        glorot_init_train_loss = [1.8601016234556451, 1.4400731981324928, 1.1124919536438642, 0.897167160486312,
                                  0.7593353841920966, 0.6672244247053242, 0.6021454189180506, 0.553894626041198,
                                  0.5167791770125256, 0.4873786554075815]
        plt.figure()
        plt.plot(range(10), zero_init_train_loss, 'b', label='Zero initialization')
        plt.plot(range(10), normal_init_train_loss, 'r', label='Normal initialization')
        plt.plot(range(10), glorot_init_train_loss, 'g', label='Glorot initialization')
        plt.xlabel('epoch')
        plt.ylabel('training loss')
        plt.title('Average training loss using different weight initializations')
        plt.legend()
        plt.savefig('initializations.jpg')

data = load_mnist()
"""
hyper_parameters={"model_architecture":[(512,512), (512,256)],
                  "learning_rate":[0.01,0.001], "batch_size":[16]}
                  #"non_linearity":['relu','tanh']}
for model_architecture in hyper_parameters["model_architecture"]:
    for learning_rate in hyper_parameters["learning_rate"]:
        for batch_size in hyper_parameters["batch_size"]:
            print(f'Results of the model_architecture= {model_architecture} , learning_rate={learning_rate}, batch_size={batch_size} ')
            nn = NN(hidden_dims=model_architecture, lr=learning_rate, batch_size=batch_size, data=data)
            train_logs = nn.train_loop(10)
"""
nn= SuperNN(data=data)
#train_logs = nn.train_loop(10)
nn.plot()
