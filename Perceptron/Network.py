
import random
import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Network(object):
    def cost(T, Y):
        tot = T * np.log(Y)
        return -tot.sum()

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a


    def train(self, training_data, epochs = 100, eta = 0.5):
        num_epochs = []
        errors = []
        n = len(training_data)
        for j in range(epochs):
            S = self.update(training_data, eta)
            num_epochs.append(j)
            errors.append(S)
        plt.plot(num_epochs,errors)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.show()
    def update(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        S = 0
        for x, y in mini_batch:
            sum,delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            S += float(sum)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        return S
    def backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        #forward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)


        # backward
        D = activations[-1] - y
        sum = np.abs(D)
        delta = D * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (sum,nabla_b, nabla_w)


    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))


def make_XOR(N_pts, noise=0.3):
    X1 = np.random.randn(int(N_pts/4), 2) * noise + [0,0]
    X2 = np.random.randn(int(N_pts/4), 2) * noise + [3, 3]
    X3 = np.random.randn(int(N_pts/4), 2) * noise + [0, 3]
    X4 = np.random.randn(int(N_pts/4), 2) * noise + [3, 0]

    X = np.r_[X1, X2, X3, X4]
    Y = np.r_[np.ones(int(N_pts/2)).T, np.zeros(int(N_pts/2)).T]
    return X, Y


def make_AND(N_pts, noise=0.3):
    X1 = np.random.randn(int(N_pts/4), 2) * noise + [0, 0]
    X2 = np.random.randn(int(N_pts/4), 2) * noise + [3, 0]
    X3 = np.random.randn(int(N_pts/4), 2) * noise + [0, 3]
    X4 = np.random.randn(int(N_pts/4), 2) * noise + [3, 3]

    X = np.r_[X1, X2, X3, X4]
    Y = np.r_[np.zeros(int(N_pts/4 * 3)).T, np.ones(int(N_pts/4)).T]
    return X, Y



def make_sine(N_pts, noise=0.3):
    X = np.linspace(0, 1, N_pts)
    Y = np.sin(2 * np.pi * 3 * X)
    Y = Y + np.random.randn(N_pts) * noise
    return X, Y






def print_line(x_test,y_test):
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for x,y in zip(x_test,y_test):
        if y == 1:
            x1.append(x[0])
            y1.append(x[1])
        else:
            x2.append(x[0])
            y2.append(x[1])
    plt.scatter(np.array(x1),np.array(y1) ,marker='o')
    plt.scatter(np.array(x2),np.array(y2) ,marker='x')
    plt.show()



def value(x):
    v = float(x)
    if v < 0.5:
        return 0
    return 1
def process(X,Y,layers):
    x_train,x_test,y_train,y_test=train_test_split(X,Y)
    training_data = [(x.reshape(-1,1),y) for x,y in zip(x_train,y_train)]
    model = Network(layers)
    model.train(training_data,500,0.5)
    testing_data = [(x.reshape(-1,1),y) for x,y in zip(x_test,y_test)]

    correct = 0
    total = len(x_test)
    ys = []
    for x,t in testing_data:
        y = value(model.feedforward(x))
        ys.append(y)
        if y == t:
            correct+=1
    print('{0}% accuracy in test set'.format(correct * 100 / total))
    print_line(x_test,ys)


X,Y = make_XOR(1000)
process(X,Y,[2,2,3,1])
