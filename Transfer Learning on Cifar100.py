import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tarfile
from six.moves import cPickle
import scipy.misc
import urllib

#cifar100

input_file = os.path.join("./", 'cifar-100-python.tar.gz')
tar_file = tarfile.open(input_file, 'r:gz')
file = tar_file.extractfile('cifar-100-python/train')
train = cPickle.load(file, encoding='latin1')
train_features = train['data'].reshape(train['data'].shape[0], 3, 32, 32).transpose(0,2,3,1).astype("float")/255
train_fine_labels = np.array(train['fine_labels'], dtype=np.uint8)
train = []
label = []
for i in range(len(train_fine_labels)):
    if train_fine_labels[i] == 29:
        train.append(train_features[i])
        label.append(train_fine_labels[i])

    if train_fine_labels[i] == 0:
        train.append(train_features[i])
        label.append(train_fine_labels[i])

for i in range(len(label)):
    if label[i] == 29:
        label[i] =1

Y= np.eye(10)[label]
X=train

#test
file = tar_file.extractfile('cifar-100-python/test')
train = cPickle.load(file, encoding='latin1')
train_features = train['data'].reshape(train['data'].shape[0], 3, 32, 32).transpose(0,2,3,1).astype("float")/255
train_fine_labels = np.array(train['fine_labels'], dtype=np.uint8)
train = []
label = []
for i in range(len(train_fine_labels)):
    if train_fine_labels[i] == 29:
        train.append(train_features[i])
        label.append(train_fine_labels[i])

    if train_fine_labels[i] == 0:
        train.append(train_features[i])
        label.append(train_fine_labels[i])

for i in range(len(label)):
    if label[i] == 29:
        label[i] =1

testY= np.eye(10)[label]
testX=train


n_classes = 10

#4.2
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=15.)

x = input_data(shape=[None, 32, 32, 3],
                      data_preprocessing=img_prep,
                      data_augmentation=img_aug)
#x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='image')
y = tf.placeholder(tf.float32, [None, 10], )

def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1]).astype("float")/255
    return images

def onehot_labels(labels):
    return np.eye(10)[labels]

def unpickle(file):
 with open(file, 'rb') as f:
  data = pickle.load(f,encoding='latin-1')
  return data

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


     conv1 = tf.nn.relu(tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME') + biases['bc1'])
     maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

     conv2 = tf.nn.relu(tf.nn.conv2d(maxpool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME') + biases['bc2'])
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
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.load('./my_model.tflearn')
    print('Model Loaded!')
    # fine_tune
    model.fit(X, Y, n_epoch=2, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=50,run_id='aa2')

    model.save('finetuned_model')
    print('FineTuned Model Saved!')
