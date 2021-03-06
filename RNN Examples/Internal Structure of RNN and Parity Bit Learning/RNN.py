
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import inSeq
import sys
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt #matplotlib inline
import random
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class RNN:
    def __init__(self, word_dim, hidden_dim=2, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
		#To implement
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.dU = []
        self.dV = []
        self.dW = []


def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, self.word_dim))
    # For each time step...
    for t in np.arange(T):
        # To implement : Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s]
RNN.forward_propagation = forward_propagation

def predict(self, x):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=1)
RNN.predict = predict

def calculate_total_loss(self, x, y):
    L = 0
    # For each sequence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L


def calculate_loss(self, x, y):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x, y) / N

RNN.calculate_total_loss = calculate_total_loss
RNN.calculate_loss = calculate_loss


def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
		#To Implement
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # To implement: Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
            dLdW += np.outer(delta_t, s[bptt_step-1])
            dLdU[:, x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
    return [dLdU, dLdV, dLdW]
RNN.bptt = bptt


# Performs one step of SGD.
def numpy_sgd_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    #To implement: Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
    self.dU.append(dLdU)
    self.dV.append(dLdV)
    self.dW.append(dLdW)
RNN.sgd_step = numpy_sgd_step


# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] >losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


model = RNN(2) # 2 vocab_size
x=[]
y=[]
X_seq=[]
Y_seq=[]

############################### train

X_seq = np.random.randint(2, size=1000)
# print(X_seq)
for i in range(1000):
    if i == 0:
        Y_seq.append(X_seq[0])
    else :
        xor = Y_seq[i-1] ^ X_seq[i]
        Y_seq.append(xor)
# print(Y_seq)
x = np.array([w for w in X_seq]).reshape((1,-1))
y = np.array([w for w in Y_seq]).reshape((1,-1))


losses = train_with_sgd(model,x, y, nepoch=50, evaluate_loss_after=1)
# plt.clf()
# plt.plot(model.dU[0][0], color='red', label = 'dU through time')
# plt.show()
################################ test

# with open(sys.argv[1], "r") as f:
#     content = f.read()
# array = content.split()
# x_test = np.asarray([int(w) for w in array]).reshape((1,-1))
# print(x_test)
# # x_test = list(map(int, x_test))
#
# with open(sys.argv[2], "r") as f2:
#     content2 = f2.read()
# array2= content2.split()
# y_test = np.asarray([int(w) for w in array2]).reshape((1,-1))
# print(y_test)
# # y_test = list(map(int, y_test))


x_test = [1,0,1,1]
y_test = [1,1,0,1]
print("X_test " , x_test)
print("Y_test " , y_test)
o , s = model.forward_propagation(x_test)
predictions = model.predict(x_test)
print("Y_test Labels predicted",predictions)

