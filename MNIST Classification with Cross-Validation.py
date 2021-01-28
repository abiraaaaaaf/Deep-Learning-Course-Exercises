import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from sklearn.model_selection import KFold , cross_val_score

log_path = "./graphs"
model_path = './model'
infer = False

mnist = input_data.read_data_sets("data/", one_hot = True)
train_x_all = mnist.train.images
train_y_all = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

## 5-fold crass validation

training_epochs = 200
batch_size = 100
learning_rate = 0.5

split_size = 5
n_input = 28*28
n_hidden_1 = 25 # 5 , 10 , 20 , 25
n_classes = 10

hyper_str = "hidden{}_lr{}".format(n_hidden_1, learning_rate)
total_batch = int(mnist.train.num_examples/batch_size)
# 2. BUILD MODEL
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, n_input], name = 'image')
    y = tf.placeholder(tf.float32, [None, n_classes], )

with tf.device('/cpu:0'):
    with tf.name_scope('Layers'):  #tf.zeros o baghye ghesmatha!!!
        # hidden_1 = tf.layers.dense(inputs=x,
        #                            units=n_hidden_1,
        #                            bias_initializer=tf.zeros_initializer(),
        #                            kernel_initializer=tf.zeros_initializer(),
        #                            activation=tf.nn.sigmoid,
        #                            name='dense1')
        # pred = tf.layers.dense(inputs=hidden_1,
        #                        bias_initializer=tf.zeros_initializer(),
        #                        kernel_initializer=tf.zeros_initializer(),
        #                        units=n_classes,
        #                        name='dense2')
        hidden_1 = tf.layers.dense(inputs=x,
                                   units=n_hidden_1,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_initializer= tf.random_normal_initializer(0., .1),
                                   activation=tf.nn.sigmoid,
                                   name = 'dense1')
        pred = tf.layers.dense(inputs=hidden_1,
                               bias_initializer=tf.zeros_initializer(),
                               kernel_initializer= tf.random_normal_initializer(0., .1),
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
        results = []
        kf = KFold(n_splits=split_size)
        for train_idx, val_idx in kf.split(train_x_all, train_y_all):
            train_x = train_x_all[train_idx]
            train_y = train_y_all[train_idx]
            val_x = train_x_all[val_idx]
            val_y = train_y_all[val_idx]
        # 3.2 Training Loop
        for epoch in range(training_epochs):
            epoch_accr = 0
            # 3.3 Loop over batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, batch_accr = \
                    sess.run([optimizer, accuracy],
                             feed_dict={x: batch_x,y: batch_y})
                epoch_accr += batch_accr/total_batch

            # Test

            test_accuracy, summ = \
                sess.run([accuracy,summeries],
                         feed_dict={x: mnist.test.images,
                                    y: mnist.test.labels})
            print("Epoch:", epoch, "Train Accuracy",
                  epoch_accr,"Test Accuracy", test_accuracy)
            writer.add_summary(summ, global_step = epoch+1)
        results.append(sess.run(accuracy, feed_dict={x: val_x, y: val_y}))
        # Save the model
        saver = tf.train.Saver()
        saver.save(sess,model_path + '/mlp_' + hyper_str + '.cpkl')
        # Close TensorBoard Writer
        writer.flush()
        writer.close()
    else:
        saver.restore(sess,model_path + '/mlp_' + hyper_str + '.cpkl')
        rand_num = np.random.randint(len(mnist.test.images))
        image = mnist.test.images[rand_num].reshape([1, 784])
        target = mnist.test.labels[rand_num]
        pred = sess.run([pred], feed_dict={x: image})
        print('True label:',np.argmax(target), 'Predicted label:',np.argmax(pred))
        imgplot = plt.imshow(image.reshape([28,28]))
        plt.show()


print( "Cross-validation result: %s", results)
