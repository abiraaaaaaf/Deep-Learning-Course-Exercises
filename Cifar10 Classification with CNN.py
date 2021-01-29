import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import dropout,input_data
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import urllib
#4.1
def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1]).astype("float")/255
    return images

def onehot_labels(labels):
    return np.eye(10)[labels]

def unpickle(file):
 with open(file, 'rb') as f:
  data = pickle.load(f,encoding='bytes')
  return data


data1 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_1")
data2 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_2")
data3 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_3")
data4 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_4")
data5 = unpickle("./cifar-10-python/cifar-10-batches-py/data_batch_5")
test1 = unpickle("./cifar-10-python/cifar-10-batches-py/test_batch")
meta = unpickle("./cifar-10-python/cifar-10-batches-py/batches.meta")

X = np.concatenate((data1[b'data'],
                    data2[b'data'],
                    data3[b'data'],
                    data4[b'data'],
                    data5[b'data']),axis=0)
X = get_proper_images(X)
Y = np.concatenate((data1[b'labels'],
                    data2[b'labels'],
                    data3[b'labels'],
                    data4[b'labels'],
                    data5[b'labels']), axis=0)

Y2 = np.array(Y)
Y = onehot_labels(Y2)
X_test = get_proper_images(test1[b'data'])
Y_test = onehot_labels(test1[b'labels'])

plt.imshow(X[20])
plt.imshow(X_test[20])
plt.show()
import tflearn
import numpy as np
import matplotlib.pyplot as plt
import six


def input_data(shape=None, placeholder=None, dtype=tf.float32,
               data_preprocessing=None, data_augmentation=None,
               name="InputData"):
    if placeholder is None:
        if shape is None:
            raise Exception("Either a `shape` or `placeholder` argument is required to consruct an input layer.")

        if len(shape) > 1 and shape[0] is not None:
            shape = list(shape)
            shape = [None] + shape

        # Create a new tf.placeholder with the given shape.
        with tf.name_scope(name):
            placeholder = tf.placeholder(shape=shape, dtype=dtype, name="X")

    tf.add_to_collection(tf.GraphKeys.INPUTS, placeholder)
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, placeholder)

    tf.add_to_collection(tf.GraphKeys.DATA_PREP, data_preprocessing)
    tf.add_to_collection(tf.GraphKeys.DATA_AUG, data_augmentation)

    return placeholder

def display_convolutions(model, layer, padding=4, filename=''):
    if isinstance(layer, six.string_types):
        vars = tflearn.get_layer_variables_by_name(layer)
        variable = vars[0]
    else:
        variable = layer.W

    data = model.get_weights(variable)

    # N is the total number of convolutions
    N = data.shape[2] * data.shape[3]

    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[0]
    # Size of the result image including padding
    result_size = filters_per_row * (filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[3]):
        for c in range(data.shape[2]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    result[filter_y * (filter_size + padding) + i, filter_x * (filter_size + padding) + j] = \
                        data[i, j, c, n]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)

    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap='gray', interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    plt.show()
#4.2

with tf.name_scope('input_data'):
    x = input_data(shape=[None, 32, 32, 3], placeholder=None, dtype='float',
               data_preprocessing=None, data_augmentation=None,
               name="image")
    #x = tf.placeholder('float', [None, 32, 32, 3], name='image')
    y = tf.placeholder('float', [None, 10], )
with tf.name_scope('Layers'):
    weights = {
                'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=0.1)),
                'wc2': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1)),
                'wd1': tf.Variable(tf.truncated_normal([8 * 8 * 64, 512], stddev=0.1)),
                'out': tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
            }

    biases = {
                'bc1': tf.Variable(tf.constant(0.1, shape=[64])),
                'bc2': tf.Variable(tf.constant(0.1, shape=[64])),
                'bd1': tf.Variable(tf.constant(0.1, shape=[512])),
                'out': tf.Variable(tf.constant(0.1, shape=[10]))
             }
#

    conv1 = tf.nn.relu(tf.nn.conv2d(x, weights['wc1'], strides=[1, 1,1, 1], padding='SAME')+biases['bc1'])
    maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.relu(tf.nn.conv2d(maxpool1, weights['wc2'], strides=[1, 1,1, 1], padding='SAME')+biases['bc2'])
    maxpool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    fc1 = tf.reshape(maxpool2, [-1, 8 * 8 * 64])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    logit = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    network = tf.nn.dropout(logit, 0.5)
    network = tf.nn.softmax(network)

    network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

with tf.device('cpu:0'):
    flag = 0
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=2, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=50 , run_id='aa2')
    #4.3
    if flag:
        display_convolutions(model, conv1, padding=4, filename='convolution1_filter')
        display_convolutions(model, conv2, padding=4, filename='convolution2_filter')

    # Save model
    model.save('my_model.tflearn')
    print('model Saved')
