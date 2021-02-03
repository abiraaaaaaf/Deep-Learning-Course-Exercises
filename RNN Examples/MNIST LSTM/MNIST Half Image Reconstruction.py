from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/mnist", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
#learning_rate = 0.001
training_steps = 3000
batch_size = 128
display_step = 10


# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_output = 28 # MNIST total classes (0-9 digits)
test_len = 128

# tf Graph input
X = tf.placeholder(tf.float32, [batch_size, None, num_output])
Y = tf.placeholder(tf.float32 , [batch_size, None , num_output])
test_output = tf.placeholder(tf.float32, [test_len, None, num_output])

# Define weights
weights = tf.get_variable('w', [num_hidden, num_output], initializer=tf.truncated_normal_initializer)
biases = tf.get_variable('b', [num_output], initializer=tf.constant_initializer(0.))
lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

def RNN(x, weights, biases):

    inputs_series = tf.unstack(x, timesteps , 1)
    # inputs_series_test = tf.unstack(test_x, timesteps, 1)
    output_series = []
    # rnn_outputs_test = []
    init_state1 = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    output_series, final_state = rnn.static_rnn(lstm_cell, inputs_series, initial_state= init_state1, dtype=tf.float32)
    reshaped_output_series= tf.reshape(tensor=output_series, shape=(-1, num_hidden))
    logits_series = tf.matmul(reshaped_output_series , weights) + biases


    return logits_series

logits_series = RNN(X,weights, biases)

regularizer = tf.nn.l2_loss(weights)
reshaped_logits = tf.reshape(tensor= logits_series, shape=(-1,1))
reshaped_output = tf.reshape(tensor=Y , shape=(-1,1))
#loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=reshaped_rnn_input, logits=reshaped_out)
loss_op = tf.reduce_mean(tf.abs(reshaped_logits - reshaped_output)) + 0.01 * regularizer

step = tf.Variable(0, trainable=False) ###?????
learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=step, decay_steps=10000, decay_rate=0.8)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Start training

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
# o = np.zeros()
for step in range(1, training_steps+1):
    #To implement training
    #_output = np.zeros((batch_size, timesteps , num_output))
    loss = 0
    batch_x, _ = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, timesteps, num_output))
    batch_y = batch_x
    batch_y[:,[0,27],:] = batch_y[:,[27,0],:]
    sess.run(train_op,feed_dict={X: batch_x, Y: batch_y})
    #if step % display_step == 0:
    lss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
    print("Epoch = %d, Loss = %f " % (step , lss))

print("Optimization Finished!")
# model test

# Calculate accuracy for 128 mnist test images

test_data = mnist.test.images[:test_len].reshape((-1, 28, 28))
# test_label = mnist.test.labels[0]
test = test_data
half_low = np.zeros((14, 28))
half_up = np.ones((14, 28))
mask = np.concatenate((half_up, half_low), axis=0)
mask = mask.reshape((-1, 28, 28))
test_out = test_data
test_data = np.multiply(mask, test_data)
test_out[:, [0, 27], :] = test_out[:, [27, 0], :]
o = sess.run(test_output, feed_dict={X:test_data , test_output:test_out})
print(o.shape)
# plt.imshow((test_data), axis=0,cmap ='grap')
plt.imshow(np.concatenate((test_data[0,:13,:],o[0,14:,:]),axis=0), cmap = 'gray')
plt.savefig("image.png")
plt.show()


