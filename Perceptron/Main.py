from Perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def make_OR(N_pts, noise=0.3):
    X1 = np.random.randn(int(N_pts/4), 2) * noise + [0, 0]
    X2 = np.random.randn(int(N_pts/4), 2) * noise + [3, 0]
    X3 = np.random.randn(int(N_pts/4), 2) * noise + [0, 3]
    X4 = np.random.randn(int(N_pts/4), 2) * noise + [3, 3]

    X = np.r_[X1, X2, X3, X4]
    Y = np.r_[np.zeros(int(N_pts/4)).T, np.ones(int(N_pts/4 * 3)).T]
    return X, Y



def make_AND(N_pts, noise=0.3):
    X1 = np.random.randn(int(N_pts/4), 2) * noise + [0, 0]
    X2 = np.random.randn(int(N_pts/4), 2) * noise + [3, 0]
    X3 = np.random.randn(int(N_pts/4), 2) * noise + [0, 3]
    X4 = np.random.randn(int(N_pts/4), 2) * noise + [3, 3]

    X = np.r_[X1, X2, X3, X4]
    Y = np.r_[np.zeros(int(N_pts/4 * 3)).T, np.ones(int(N_pts/4)).T]
    return X, Y




def print_line(weights,x_test,y_test):
    x  = np.linspace(-3,3,100)
    m = - weights[0] / weights[1]
    b = - weights[2] / weights[1]
    y = m*x + b
    plt.plot(x,y,'-r')
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


def process(X,Y):
    x_train,x_test,y_train,y_test=train_test_split(X,Y)
    model = Perceptron(x_train,y_train,100)
    predictions = model.predictions(x_test)
    total = len(x_test)
    correct = (predictions == y_test).sum()
    print('{0}% accuracy in test set'.format(correct * 100 / total))
    print("Weights")
    print(model.weights[:-1])
    print("Threshold")
    print(model.weights[-1])
    print_line(model.weights,x_test,y_test)





X,Y = make_AND(100)
process(X,Y)


X,Y = make_OR(100)
process(X,Y)
