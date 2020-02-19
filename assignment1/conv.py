import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_mnist():
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    print("max(train_data[0]) ", np.max(train_data[0]))
    print(train_data[0].shape)
    train_inputs = [np.reshape(x, (1, 28, 28)) for x in train_data[0]]
    train_data = np.array(train_inputs).reshape(-1, 1, 28, 28), np.array(train_data[1])

    val_inputs = [np.reshape(x, (1, 28, 28)) for x in val_data[0]]
    val_data = np.array(val_inputs).reshape(-1, 1, 28, 28), np.array(val_data[1])

    test_inputs = [np.reshape(x, (1, 28, 28)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return train_data, val_data, test_data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        # x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        return x


net = Net()
train_data, val_data, test_data = load_mnist()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

def train_loop(n_epochs, batch_size, train_data, valid_data):
    X_train, y_train = train_data

    X_train = torch.Tensor(X_train)
    y_train = torch.LongTensor(y_train)

    X_valid, y_valid = valid_data
    X_valid = torch.Tensor(X_valid)
    y_valid = torch.LongTensor(y_valid)

    train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

    n_batches = int(np.ceil(X_train.shape[0] / batch_size))
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            minibatchX = X_train[batch_size * batch:batch_size * (batch + 1), :]
            minibatchY = y_train[batch_size * batch:batch_size * (batch + 1)]

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(minibatchX)
            loss = criterion(outputs, minibatchY)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = compute_loss_and_accuracy(X_train, y_train)
        valid_loss, valid_accuracy = compute_loss_and_accuracy(X_valid, y_valid)

        print("Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss",
              epoch, train_accuracy, valid_accuracy, train_loss, valid_loss)
        train_logs['train_accuracy'].append(train_accuracy)
        train_logs['validation_accuracy'].append(valid_accuracy)
        train_logs['train_loss'].append(train_loss)
        train_logs['validation_loss'].append(valid_loss)

    return train_logs


def compute_loss_and_accuracy(X, y):
    outputs = net(X)
    loss = criterion(outputs, y)

    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y).sum().item()
    accuracy = correct / len(y)
    return loss.detach(), accuracy

#train_loop(10, 16, train_data, val_data)

"""
CNN results, no dropout
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 0 0.9833 0.9833 tensor(0.0527) tensor(0.0573)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 1 0.98996 0.9877 tensor(0.0312) tensor(0.0460)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 2 0.99262 0.9899 tensor(0.0232) tensor(0.0396)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 3 0.99416 0.9897 tensor(0.0188) tensor(0.0443)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 4 0.99514 0.9896 tensor(0.0157) tensor(0.0485)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 5 0.9951 0.9886 tensor(0.0143) tensor(0.0518)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 6 0.9965 0.9899 tensor(0.0108) tensor(0.0448)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 7 0.9975 0.9912 tensor(0.0078) tensor(0.0432)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 8 0.9942 0.9885 tensor(0.0196) tensor(0.0539)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 9 0.99768 0.9904 tensor(0.0069) tensor(0.0467)


CNN results, with dropout
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 0 0.97792 0.9805 tensor(0.0717) tensor(0.0694)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 1 0.98472 0.9852 tensor(0.0474) tensor(0.0528)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 2 0.99098 0.9887 tensor(0.0281) tensor(0.0439)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 3 0.99318 0.9892 tensor(0.0215) tensor(0.0394)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 4 0.99284 0.9891 tensor(0.0216) tensor(0.0378)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 5 0.99378 0.9894 tensor(0.0199) tensor(0.0382)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 6 0.99628 0.9927 tensor(0.0117) tensor(0.0300)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 7 0.99588 0.9918 tensor(0.0127) tensor(0.0344)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 8 0.99642 0.9908 tensor(0.0111) tensor(0.0394)
Epoch, train_accuracy, valid_accuracy, train_loss, valid_loss 9 0.99606 0.992 tensor(0.0121) tensor(0.0317)
"""



def plot():
    cnn_train_loss = [0.0527, 0.0312, 0.0232, 0.0188, 0.0157, 0.0143, 0.0108, 0.0078, 0.0196, 0.0069]
    cnn_val_loss = [0.0573, 0.0460, 0.0396, 0.0443, 0.0485, 0.0518, 0.0448, 0.0432, 0.0539, 0.0467]

    """
    mlp_train_loss = [0.28153537862947453, 0.20143843526836797, 0.15624305943113298, 0.12668418587895094,
                      0.10554932123361933, 0.08997433320011995, 0.0776907221059160, 0.0678381166880755,
                      0.059775911369792066, 0.05302756704624103]
    mlp_val_loss = [0.2595818258216654, 0.1924245873136336, 0.15549514070045584, 0.13261075078818138,
                    0.11730770919579483, 0.10693320238744197, 0.09922512335869743,
                    0.09333125136136562, 0.0887679412296058, 0.08512650380991808]
    """
    dropout_cnn_train_loss = [0.0717, 0.0474, 0.0281, 0.0215, 0.0216, 0.0199, 0.0117, 0.0127, 0.0111, 0.0121]
    dropout_cnn_val_loss = [0.0694, 0.0528, 0.0439, 0.0394, 0.0378, 0.0382, 0.0300, 0.0344, 0.0394, 0.0317]

    plt.figure()
    plt.plot(range(10), cnn_train_loss, 'b', label='CNN train')
    plt.plot(range(10), cnn_val_loss, 'r', label='CNN val')
    plt.plot(range(10), dropout_cnn_train_loss, 'g', label='DropoutCNN train')
    plt.plot(range(10), dropout_cnn_val_loss, 'k', label='DropoutCNN valid')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Average training and validation loss for Vanilla-CNN and Dropout-CNN')
    plt.legend()
    plt.savefig('cnn_drop.jpg')



plot()