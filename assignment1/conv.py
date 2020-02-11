import numpy as np
import pickle
import gzip

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
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(x)
        return x



net = Net()
train_data, val_data, test_data = load_mnist()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
    return loss, accuracy

train_loop(10, 64, train_data, val_data)



