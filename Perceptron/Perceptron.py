
import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, inputs, labels, epochs=100, learning_rate=0.5):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(len(inputs[0]) + 1) ##including threshold
        self.train(inputs,labels)


    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))


    def train(self,inputs,labels):
        num_epochs = []

        errors = []
        for _ in range(self.epochs):
            error_sum = 0
            for input, label in zip(inputs, labels):
                input = np.append(input,1)
                summation = np.dot(input, self.weights)

                prediction = self.sigmoid(summation)
                error_sum += np.abs(label - prediction)
                prediction_prime = self.sigmoid_prime(summation)
                self.weights += self.learning_rate * (label - prediction) * input * prediction_prime
            num_epochs.append(_)
            errors.append(error_sum)
        plt.plot(num_epochs,errors)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.show()


    def predict_single(self,x):
        input = x.copy()
        input = np.append(input,1)
        summation = np.dot(input,self.weights)
        value = self.sigmoid(summation)
        if value >= 0.5:
            return 1
        return 0
    def predictions(self,inputs):
        y = []
        for i in inputs:
            y.append(self.predict_single(i))
        return np.array(y)
