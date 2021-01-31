from __future__ import print_function
import string
import zipfile
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import rnn_cell_impl

N = 10

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return (1/1+np.exp(-x))

def call2(input, c,h, weights, bias):
    """Long short-term memory cell (LSTM)."""
    x = np.concatenate([input , h])

    res = np.matmul(x, weights)
    concat = res + bias

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i = sigmoid(concat[:10])
    j = np.tanh(concat[10:20])
    f = sigmoid(concat[20:30])
    o = sigmoid(concat[30:])

    new_c = np.multiply(c,f) + np.multiply(i,j)
    new_h = np.multiply(np.tanh(new_c) , o)

    return new_h, new_c, i, j, f, o

class BatchGenerator:

    def __init__(self, text, vocabulary, batch_size, num_unrollings):
        self.text = text
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.num_unrollings = num_unrollings
        segment_size = len(text) // batch_size
        self.cursor = [i * segment_size for i in range(batch_size)]
        self.last_batch = self.next_batch()

    def next_batch(self):
        batch = np.zeros((self.batch_size, len(self.vocabulary)))
        for b in range(self.batch_size):
            c = self.text[self.cursor[b]]
            batch[b, self.vocabulary.index(c)] = 1.0
            self.cursor[b] = (self.cursor[b] + 1) % len(self.text)
        return batch

    def next(self):
        batches = [self.last_batch]
        for step in range(self.num_unrollings):
            batches.append(self.next_batch())
        self.last_batch = batches[-1]
        return np.asarray(batches)

class LSTM:

    def __init__(self, NUM_VOC):
        self.inputs = tf.placeholder(tf.float32, (None, None, NUM_VOC))
        self.outputs = tf.placeholder(tf.float32, (None, None, NUM_VOC))

        cell = tf.contrib.rnn.BasicLSTMCell(N, state_is_tuple=True)

        rnn_outputs, rnn_states= tf.nn.dynamic_rnn(
            cell, self.inputs, dtype=tf.float32, time_major=True)
###############
        self.xx = cell.weights[0]
        self.yy = cell.weights[1]
        # print(self.xx)
        tf.summary.scalar("rnn_states", rnn_states[0][1,1])

##############
        raw_outputs = layers.fully_connected(rnn_outputs, NUM_VOC, activation_fn=None)
        self.prediction = tf.nn.softmax(raw_outputs)
        self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=raw_outputs, labels=self.outputs))
        self.train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.error)

def main():
    text = tf.compat.as_str(zipfile.ZipFile('text8.zip').read('text8.txt'))
    print(text)
    VOCABULARY = 'abx'
    NUM_VOC = len(VOCABULARY)

    train_batches = BatchGenerator(text, VOCABULARY, 5, N) # batch_size = 6 , N

    sess = tf.Session()
    lstm = LSTM(NUM_VOC)
    merged_summary_op = tf.summary.merge_all() #
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("output", sess.graph) #

    NUM_EPOCH = 30
    NUM_ITER = 10
    errs = []
#################################
    file1 = open("Hidden", "w")
    file2 = open("Cell_State", "w")
    file5 = open("Forget_Gate", "w")
    file3 = open("Input_Gate", "w")
    file6 = open("Output_Gate", "w")
    file4 = open("State", "w")
###############################
    for i in range(1):

        for step in range(1, NUM_EPOCH * NUM_ITER + 1):
            batches = train_batches.next()
            inputs = batches[:N, :, :]
            outputs = batches[1:, :, :]
####
            err,_,w,b, summary = sess.run([lstm.error, lstm.train, lstm.xx, lstm.yy,merged_summary_op],
                              feed_dict={lstm.inputs: inputs,lstm.outputs: outputs}) # weight cell lstm o be onvane khorooji bar migardoone
            errs.append(err)
###########
            info = call2([1, 0, 0], np.zeros((1, 1)), np.zeros((1, 10))[0], w, b)
            file1.write(str(info[0]) + "\n")   #h_new
            file2.write(str(info[1]) + "\n")    #c_new
            file3.write(str(info[2]) + "\n")    #input
            file4.write(str(info[3]) + "\n")    #j
            file5.write(str(info[4]) + "\n")    #forget
            file6.write(str(info[5]) + "\n")    #output

            writer.add_summary(summary, step)
            ######
            if step % NUM_ITER == 0:
                print('epoch:', step / NUM_ITER, 'avg err:', sum(errs, 0) / len(errs))
                errs = []
                # random generation
                inputs = np.zeros((1, 1, NUM_VOC))
                idx = np.random.randint(0, NUM_VOC)
                inputs[0, 0, idx] = 1.0
                for i in range(20):     # predict and concat to the input for the next lstm input
                    pred = sess.run(lstm.prediction, feed_dict={lstm.inputs: inputs[-10:, :, :]})
                    idx = np.random.choice(NUM_VOC, p=pred[-1, 0])
                    newcol = np.zeros((1, 1, NUM_VOC))
                    newcol[0, 0, idx] = 1.0
                    inputs = np.concatenate([inputs, newcol], axis=0)
                s = ''
                for i in range(inputs.shape[0]):
                    s += VOCABULARY[np.argmax(inputs[i, 0])]
                print('gen:', repr(s))
################################

    # q2 part2

    file = open("Cell_State_Neurons", "w")
    for i in range(15):
        info = call2([1, 0, 0], np.zeros((1 , 1)), np.zeros((1, 10))[0], w, b)
        file.write(str(info[1][0])+"\n")
    info = call2([0, 0, 1], np.zeros((1, 1)), np.zeros((1, 10))[0], w, b)
    file.write(str(info[1][0])+"\n")
    for i in range(15):
        info = call2([0, 1, 0], np.zeros((1, 1)), np.zeros((1, 10))[0], w, b)
        file.write(str(info[1][0])+"\n")

    file.close()
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    file6.close()

#############################
if __name__ == '__main__':
    main()

