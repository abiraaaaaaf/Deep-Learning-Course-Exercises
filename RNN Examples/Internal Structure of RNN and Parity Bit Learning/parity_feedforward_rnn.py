import numpy as np
import sys
import tensorflow as tf
import itertools
from tensorflow.contrib import rnn

def logical_xor(str1, str2):
    return bool(str1) ^ bool(str2)

# input output train

x=[]
y=[]
x_test=[]
xseq=[[]]


with open("text1.5_input.txt", "r") as f:
    xseq = f.read().splitlines()
with open("text1.5_output.txt", "r") as f:
    yseq = f.read().splitlines()

# X_seq = np.array(xseq)
# Y_seq = np.array(yseq)
# X_seq.reshape(2000,-1)
#
# file = open(sys.argv[1], "r")
# x_test = file.read().splitlines()
# file = open(sys.argv[2], "r")
# y_test = file.read().splitlines()
# # x_test = x_test.split(',')
# x_test = np.array(x_test)
# x_test.reshape(20,-1)

training_epochs = 10000
batch_size = 8
learning_rate = 0.01
n_input = 6 # 5
n_hidden_rnn = 128
n_out = 6 # 5


xx = list( itertools.product([0, 1], repeat=n_input))
yy = [[np.count_nonzero(i[:j + 1]) % 2 for j in range(len(i))] for i in xx]
total_batch = int(len(xx)/batch_size)

def RNN(x, weights, biases, n_input, n_hidden_rnn):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_input, 1)
    rnn_cell = rnn.LSTMCell(n_hidden_rnn)
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.add(tf.matmul(outputs[-1], weights['w']), biases['b'])


with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, n_input], name = 'text')
    y = tf.placeholder(tf.float32, shape=[None, n_out], name='output')
    l_rate = tf.placeholder(tf.float32, shape=[],name='learning_rate')

w_rnn = {
        'w': tf.Variable(tf.random_normal([n_hidden_rnn, n_out]))
    }
b_rnn = {
        'b': tf.Variable(tf.random_normal([n_out]))
}

with tf.device('/cpu:0'):
    with tf.name_scope('Layers'):
        rnn_out = RNN(x, w_rnn, b_rnn, n_input, n_hidden_rnn)
    with tf.name_scope('Loss_Optimizer'):
        cost = tf.reduce_mean(tf.nn.l2_loss((y - rnn_out)))
        optimizer1 = tf.train. \
            AdamOptimizer(learning_rate=learning_rate)\
            .minimize(cost)
    with tf.name_scope('Accuracy'):
        correct_pred1 = tf.equal(tf.argmax(rnn_out, 1), tf.argmax(y, 1))
        accuracy1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))

with tf.Session() as sess:
    # 3.0 Initialize all variables
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        epoch_accr = 0
        epoch_loss = 0
        if epoch >= 5000 and i % 5000 == 0:
            learning_rate = learning_rate / 10
        # 3.3 Loop over batches
        for i in range(total_batch):
            batch_x, batch_y = xx[i*8:(i+1)*8], yy[i*8:(i+1)*8]
            batch_loss , batch_accr,_  = \
                sess.run([cost ,accuracy1,optimizer1],
                         feed_dict={x: batch_x, y: batch_y, l_rate:learning_rate})

            epoch_accr += batch_accr / total_batch
            epoch_loss += batch_loss / total_batch
        print("Epoch:", epoch, "Train Accuracy",
              epoch_accr, "Train Loss", epoch_loss)



