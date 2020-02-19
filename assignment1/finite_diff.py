import pickle
import numpy as np
import gzip
import matplotlib.pyplot as plt

def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]

def load_mnist():
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    print("max(train_data[0]) ", np.max(train_data[0]))
    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [one_hot(y, 10) for y in train_data[1]]
    train_data = np.array(train_inputs).reshape(-1, 784), np.array(train_results).reshape(-1, 10)

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [one_hot(y, 10) for y in val_data[1]]
    val_data = np.array(val_inputs).reshape(-1, 784), np.array(val_results).reshape(-1, 10)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return train_data, val_data, test_data

# train_data_, val_data_, test_data_ = load_mnist()

class NN(object):
    def __init__(self,
                 hidden_dims=(784, 256),
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=64,
                 seed=None,
                 activation="relu",
                 data=None
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        #self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if data is None:
            # for testing, do NOT remove or modify
            self.train, self.valid, self.test = (
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400)))
        )
        else:
            self.train, self.valid, self.test = data


    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            sd = np.sqrt(6.0 / (all_dims[layer_n-1] + all_dims[layer_n]))
            self.weights[f"W{layer_n}"] = np.random.uniform(-sd, sd, (all_dims[layer_n-1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            return np.where(x <= 0, 0, 1)
        return np.maximum(0, x)

    def sigmoid(self, x, grad=False):
        if grad:
            return (1.0/(1.0+np.exp(-x)))*(1.0-1.0/(1.0+np.exp(-x)))
        return 1.0/(1.0+np.exp(-x))

    def tanh(self, x, grad=False):
        if grad:
            return 1-(np.tanh(x)*np.tanh(x))
        return np.tanh(x)

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.relu(x, grad)
            pass
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
            pass
        else:
            raise Exception("invalid")

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        #e_x = np.exp(x - np.max(x))
        if len(x.shape) > 1:
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)
        else:
            e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return e_x / e_x.sum(axis=0, keepdims=True)

    def forward(self, x, print_flag=False, i=0, j=0):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i

        for layer_n in range(1, self.n_hidden+1):

            cache[f"A{layer_n}"] = np.matmul(cache[f"Z{layer_n-1}"], self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])

            """
            if print_flag:
                if layer_n == 2:
                    print("weights[fW{layer_n}][:10] ", weights[f"W{layer_n}"][i][j])
                    print("cache[fZ{layer_n-1}]", cache[f"Z{layer_n-1}"])
                    print("cache[fA{layer_n}] ", cache[f"A{layer_n}"])
                    print("cache[fZ{layer_n}] ", cache[f"Z{layer_n}"])
            """

            """
            print("layer_n", layer_n)
            print("cache[fZ{layer_n-1}] ", cache[f"Z{layer_n-1}"][0])
            print("self.weights[fW{layer_n}] ", self.weights[f"W{layer_n}"][0])
            print("self.weights[fb{layer_n}] ", self.weights[f"b{layer_n}"][0])
            print("cache[fA{layer_n}]", cache[f"A{layer_n}"][0])
            print("cache[fZ{layer_n}]", cache[f"Z{layer_n}"][0])
            """
            assert not np.any(np.isnan(self.weights[f"W{layer_n}"]))
            assert not np.any(np.isnan(self.weights[f"b{layer_n}"]))
            assert not np.any(np.isnan(cache[f"A{layer_n}"]))
            assert not np.any(np.isnan(cache[f"Z{layer_n}"]))


        cache[f"A{self.n_hidden + 1}"] = np.matmul(cache[f"Z{self.n_hidden }"], self.weights[f"W{self.n_hidden+1}"]) \
                                         + self.weights[f"b{self.n_hidden+1}"]
        cache[f"Z{self.n_hidden + 1}"] = self.softmax(cache[f"A{self.n_hidden + 1}"])
        """
        print("output layer")
        print("cache[fZ{self.n_hidden + 1}] ", cache[f"Z{self.n_hidden}"][0])
        print("self.weights[fW{self.n_hidden + 1}] ", self.weights[f"W{self.n_hidden + 1}"][0])
        print("self.weights[fb{self.n_hidden + 1}] ", self.weights[f"b{self.n_hidden + 1}"][0])
        print("cache[fA{self.n_hidden + 1}]", cache[f"A{self.n_hidden + 1}"])
        print("cache[fZ{self.n_hidden + 1}]", cache[f"Z{self.n_hidden + 1}"])
        """
        assert not np.any(np.isnan(self.weights[f"W{self.n_hidden + 1}"]))
        assert not np.any(np.isnan(self.weights[f"b{self.n_hidden + 1}"]))
        assert not np.any(np.isnan(cache[f"A{self.n_hidden + 1}"]))
        assert not np.any(np.isnan(cache[f"Z{self.n_hidden + 1}"]))

        return cache[f"Z{self.n_hidden + 1}"], cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1

        # number of examples
        n = len(labels)

        grads[f"dA{self.n_hidden + 1}"] = output - labels

        grads[f"dW{self.n_hidden + 1}"] = (1. / n) * \
                                          np.sum(np.matmul(
                                            np.expand_dims(cache[f"Z{self.n_hidden}"], axis=-1),
                                            np.expand_dims(grads[f"dA{self.n_hidden + 1}"], axis=-2)
                                          ), axis=0)
        grads[f"db{self.n_hidden + 1}"] = (1. / n) * np.sum(grads[f"dA{self.n_hidden + 1}"], axis=0, keepdims=True)

        for layer_n in range(self.n_hidden, 0, -1):
            grads[f"dZ{layer_n}"] = np.dot(grads[f"dA{layer_n+1}"], self.weights[f"W{layer_n+1}"].T)

            grads[f"dA{layer_n}"] = grads[f"dZ{layer_n}"] * \
                                           self.activation(cache[f"A{layer_n}"], grad=True)

            grads[f"dW{layer_n}"] =(1. / n) * \
                                   np.sum(np.matmul(
                np.expand_dims(cache[f"Z{layer_n - 1}"], axis=-1),
                np.expand_dims(grads[f"dA{layer_n}"], axis=-2),
                                   ), axis=0)

            grads[f"db{layer_n}"] = (1. / n) * np.sum(grads[f"dA{layer_n}"], axis=0, keepdims=True)

        return grads


    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - self.lr * grads[f"db{layer}"]


    # def one_hot(self, y, n_classes=None):
    #     n_classes = n_classes or self.n_classes
    #     return np.eye(n_classes)[y]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        loss = np.mean(- np.sum(labels * np.log(prediction), axis=1))
        return loss

    def compute_loss_and_accuracy(self, X, y):
        one_y = y
        y = np.argmax(y, axis=1)  # Change y to integers
        _, cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        rnd_example = X_train[2:3]
        rnd_label = y_train[2:3]

        for epoch in range(n_epochs):
            predictions, cache = self.forward(rnd_example)
            # Backward
            grads = self.backward(cache, rnd_label)
            # Update
            self.update(grads)
            print("Loss ", self.loss(predictions, rnd_label))


        _, cache = self.forward(rnd_example)
        grads = self.backward(cache, rnd_label)
        N_values = [k * 10 ** i for i in range(0, 5) for k in range(1, 6)]
        print(len(N_values))
        layer = 2
        print("Layer ", layer)
        max_diff = []
        for N in N_values:
            eps = 1.0 / N
            delta = []
            for i, (W, dW) in enumerate(zip(self.weights["W" + str(layer)], grads["dW" + str(layer)])):
                if i > 0:
                    break
                for j, (w, dw) in enumerate(zip(W, dW)):
                    if j>10:
                        break

                    self.weights["W" + str(layer)][i][j] = w + eps
                    output1, _ = self.forward(rnd_example)
                    loss1 = self.loss(output1, rnd_label)

                    self.weights["W" + str(layer)][i][j] = w - eps
                    output2, _ = self.forward(rnd_example)
                    loss2 = self.loss(output2, rnd_label)

                    self.weights["W" + str(layer)][i][j] = w

                    est_grad = (loss1 - loss2)  * N /2
                    delta.append(abs(est_grad - dw))
            max_diff.append(np.max(delta))
            print("N, Max diff ", N, np.max(delta))

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy

    """
    N, Max diff 1 0.006544506913660486
    N, Max diff  2 0.005216486602912074
    N, Max diff  3 0.003915086286613523
    N, Max diff  4 0.0026203439370533488
    N, Max diff  5 0.0013282652305524056
    N, Max diff  10 6.717387589748325e-09
    N, Max diff  20 1.6793464943307135e-09
    N, Max diff  30 7.463739062560371e-10
    N, Max diff  40 4.198384409223599e-10
    N, Max diff  50 2.6869489879594033e-10
    N, Max diff  100 6.716721536598191e-11
    N, Max diff  200 1.6818601199231065e-11
    N, Max diff  300 7.448318871394743e-12
    N, Max diff  400 4.16205871850428e-12
    N, Max diff  500 2.6632576352603188e-12
    N, Max diff  1000 6.093450397037792e-13
    N, Max diff  2000 3.268028209157947e-13
    N, Max diff  3000 3.275062512853033e-13
    N, Max diff  4000 5.797541266505668e-13
    N, Max diff  5000 8.607030019258666e-13
    N, Max diff  10000 1.0543996231682229e-12
    N, Max diff  20000 2.2762694507072467e-12
    N, Max diff  30000 3.0371924593375343e-12
    N, Max diff  40000 4.767694919416421e-12
    N, Max diff  50000 6.826593977604656e-12
    """

    def plot(self):
        max_diff = [0.006544506913660486, 0.005216486602912074, 0.003915086286613523, 0.0026203439370533488,
                    0.0013282652305524056, 6.717387589748325e-09, 1.6793464943307135e-09, 7.463739062560371e-10,
                    4.198384409223599e-10, 2.6869489879594033e-10, 6.716721536598191e-11, 1.6818601199231065e-11,
                    7.448318871394743e-12, 4.16205871850428e-12, 2.6632576352603188e-12, 6.093450397037792e-13,
                    3.268028209157947e-13, 3.275062512853033e-13, 5.797541266505668e-13, 8.607030019258666e-13,
                    1.0543996231682229e-12, 2.2762694507072467e-12, 3.0371924593375343e-12, 4.767694919416421e-12,
                    6.826593977604656e-12]
        N = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000,
             10000, 20000, 30000, 40000, 50000]
        plt.figure()
        plt.plot(N, max_diff, 'b')
        plt.xscale("log")
        plt.xlabel('N')
        plt.ylabel('maximum difference')
        plt.savefig('finite_diff.jpg')

data = load_mnist()

nn = NN(hidden_dims=(512,512), lr=0.01, batch_size=16, data=data)
#train_logs  = nn.train_loop(2)
nn.plot()
