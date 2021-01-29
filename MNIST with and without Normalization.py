import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.python.framework import dtypes
import os
import numpy as np
import matplotlib.pyplot as plt
from load_mnist import MNIST
from sklearn import preprocessing

#train_x_all, train_y_all = MNIST(path="./",return_type="numpy",mode="vanilla").load_training()
#test_x, test_y = MNIST(path="./",return_type="numpy",mode="vanilla").load_testing()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, dtype=dtypes.uint8, validation_size=0)

log_path = "./graphs"
model_path = './model'
infer = False

train_x_all = mnist.train.images
train_y_all = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels


#Normalize from [0:255] => [0.0:1.0]
# train_x_all = np.multiply(train_x_all, 1.0 / 255.0)
# train_x_all_size = train_x_all.shape[1]
# train_x_all_width = train_x_all_height = np.ceil(np.sqrt(train_x_all_size)).astype(np.uint8)

for data in [test_x,train_x_all]:
    print("shape", data.shape)
    print("mean", data.mean())
    print("max",data.max())
    print("min", data.min())
    # var=np.square(data-data.mean)
    # var /= 10000
    # data=(data - data.mean)/var
def normalize1(input_data):
    average = input_data.mean(axis=0)
    total = input_data.sum(axis=0)
    variance = input_data.var(axis=0)
    normalized = (input_data - average) / variance
    return normalized

def normalize(input_data):
    minimum = input_data.min(axis=0)
    maximum = input_data.max(axis=0)
    normalized = (input_data - minimum) / ( maximum - minimum )
    #normalized = preprocessing.normalize(input_data, norm='l2')
    return normalized
# train_x_all = normalize1(train_x_all)
# train_x_all = normalize(train_x_all)

training_epochs = 4
batch_size = 60
learning_rate = 0.01
training_step = 0
n_input = 28*28
n_hidden_1 = 10
n_classes = 10

hyper_str = "hidden{}_lr{}".format(n_hidden_1, learning_rate)
total_batch = int(mnist.train.num_examples/batch_size)
# 2. BUILD MODEL
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, n_input], name = 'image')
    y = tf.placeholder(tf.float32, [None, n_classes], )

with tf.device('/cpu:0'):
    with tf.name_scope('Layers'):  #tf.zeros o baghye ghesmatha!!!

        hidden_1 = tf.layers.dense(inputs=x,
                                   units=n_hidden_1,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_initializer= tf.random_normal_initializer(0., 0.2),
                                   activation=tf.nn.sigmoid,
                                   name = 'dense1')
        pred = tf.layers.dense(inputs=hidden_1,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_initializer= tf.random_normal_initializer(0., 0.2),
                               units=n_classes,
                               name = 'dense2')

    # 2.2 Define Loss and Optimizer
    with tf.name_scope('Loss_Optimizer'):
        loss = tf.losses.softmax_cross_entropy(logits=pred,
                                               onehot_labels=y)
        optimizer = tf.train.\
            GradientDescentOptimizer(learning_rate=learning_rate)\
            .minimize(loss)

    # 2.3 Define Accuracy
    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 2.4 Create a summary for accuracy & loss:
tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)
summeries  = tf.summary.merge_all()
if not infer:
    writer = tf.summary.FileWriter(log_path+'/'+hyper_str)

# 2.5 Add op for saving the model
save_path = model_path + '/mlp_' + hyper_str + '.cpkl'
if not os.path.exists(save_path):
    os.makedirs(save_path)
saver = tf.train.Saver()
# 3. RUN THE MODEL
with tf.Session() as sess:
    # 3.0 Initialize all variables
    sess.run(tf.global_variables_initializer())
    if not infer:
        # 3.1 Use TensorBoard
        writer.add_graph(sess.graph)
        # 3.2 Training Loop
        for epoch in range(training_epochs):
            epoch_accr = 0
            # 3.3 Loop over batches
            training_step = 0
            for i in range(total_batch):
                training_step = training_step + 1
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                batch_loss , batch_accr = \
                    sess.run([optimizer, accuracy],
                             feed_dict={x: batch_x,y: batch_y})
                epoch_accr += batch_accr/total_batch

            # Test

            test_accuracy, summ = \
                sess.run([accuracy,summeries],
                         feed_dict={x: test_x,
                                    y: test_y})
            print("Epoch:", epoch, "Train Accuracy",
                  epoch_accr,"Test Accuracy", test_accuracy)
            writer.add_summary(summ, global_step = epoch+1)

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess,model_path + '/mlp_' + hyper_str + '.cpkl')
        # Close TensorBoard Writer
        writer.flush()
        writer.close()
    else:
        saver.restore(sess,model_path + '/mlp_' + hyper_str + '.cpkl')
        rand_num = np.random.randint(len(test_x))
        image = test_x[rand_num].reshape([1, 784])
        target = test_y[rand_num]
        pred = sess.run([pred], feed_dict={x: image})
        print('True label:',np.argmax(target), 'Predicted label:',np.argmax(pred))
        imgplot = plt.imshow(image.reshape([28,28]))
        plt.show()



