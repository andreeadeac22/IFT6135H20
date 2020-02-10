import pickle
import numpy as np
import gzip

def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]

def load_mnist():
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

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
            return (1/1+np.exp(-x))*(1-1/1+np.exp(-x))
            pass
        return 1/1+np.exp(-x)

    def tanh(self, x, grad=False):
        if grad:
            return 1-(np.tanh(x)*np.tanh(x))
            pass
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
        e_x = np.exp(x - np.max(x))
        if len(x.shape) > 1\
                :
            return e_x / e_x.sum(axis=1, keepdims=True)
        else:
            return e_x / e_x.sum(axis=0, keepdims=True)

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        for layer_n in range(1, self.n_hidden+1):
            cache[f"A{layer_n}"] = np.matmul(cache[f"Z{layer_n-1}"], self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])
            print("cache[fA{layer_n}]", cache[f"A{layer_n}"])
            print("cache[fZ{layer_n}]", cache[f"Z{layer_n}"])

        cache[f"A{self.n_hidden + 1}"] = np.matmul(cache[f"Z{self.n_hidden }"], self.weights[f"W{self.n_hidden+1}"]) \
                                         + self.weights[f"b{self.n_hidden+1}"]
        cache[f"Z{self.n_hidden + 1}"] = self.softmax(cache[f"A{self.n_hidden + 1}"])
        return cache

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
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]

                # Forward
                cache = self.forward(minibatchX)
                # Backward
                grads = self.backward(cache, minibatchY)
                # Update
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            print("Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss",
                  epoch, train_accuracy, valid_accuracy, train_loss, valid_loss)
            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy