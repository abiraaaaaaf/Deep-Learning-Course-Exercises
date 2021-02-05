import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
z_dim = 8
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 64
c = 0
lr = 1e-3


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28))

    return fig

#https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
c = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X, c):
    inputs = tf.concat(axis=1, values=[X, c])
    h = tf.nn.relu(tf.matmul(inputs, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    h = tf.nn.relu(tf.matmul(inputs, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X, c)
z_sample = sample_z(z_mu, z_logvar)
_, logits = P(z_sample, c)

# Sampling from random z
X_samples, _ = P(z, c)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out1/'):
    os.makedirs('out1/')
if not os.path.exists('out2/'):
    os.makedirs('out2/')
if not os.path.exists('out3/'):
    os.makedirs('out3/')
if not os.path.exists('out4/'):
    os.makedirs('out4/')
if not os.path.exists('out5/'):
    os.makedirs('out5/')

i1 = 0
i2 = 0
i3 = 0
i4 = 0
i5 = 0

for it in range(200000):
    X_mb, y_mb = mnist.train.next_batch(mb_size)

    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb, c: y_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print()

        # y = np.zeros(shape=[5, y_dim])
        # y[:, np.random.randint(0, y_dim)] = 1.
        y1 = np.zeros(shape=[5, y_dim])
        y1[:, 5] = 1.
        y2 = np.zeros(shape=[5, y_dim])
        y2[:, 6] = 1.
        y3 = np.zeros(shape=[5, y_dim])
        y3[:, 7] = 1.
        y4 = np.zeros(shape=[5, y_dim])
        y4[:, 8] = 1.
        y5 = np.zeros(shape=[5, y_dim])
        y5[:, 9] = 1.

        #samples = sess.run(X_samples, feed_dict={z: np.random.randn(5, z_dim), c: y})
        samples1 = sess.run(X_samples, feed_dict={z: np.random.randn(5, z_dim), c: y1})
        samples2 = sess.run(X_samples, feed_dict={z: np.random.randn(5, z_dim), c: y2})
        samples3 = sess.run(X_samples, feed_dict={z: np.random.randn(5, z_dim), c: y3})
        samples4 = sess.run(X_samples, feed_dict={z: np.random.randn(5, z_dim), c: y4})
        samples5 = sess.run(X_samples, feed_dict={z: np.random.randn(5, z_dim), c: y5})

        # fig = plot(samples)
        # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        # i += 1
        # plt.close(fig)
        fig = plot(samples1)
        plt.savefig('out1/{}.png'.format(str(i1).zfill(3)), bbox_inches='tight')
        i1 += 1
        plt.close(fig)
        fig = plot(samples2)
        plt.savefig('out2/{}.png'.format(str(i2).zfill(3)), bbox_inches='tight')
        i2 += 1
        plt.close(fig)
        fig = plot(samples3)
        plt.savefig('out3/{}.png'.format(str(i3).zfill(3)), bbox_inches='tight')
        i3 += 1
        plt.close(fig)
        fig = plot(samples4)
        plt.savefig('out4/{}.png'.format(str(i4).zfill(3)), bbox_inches='tight')
        i4 += 1
        plt.close(fig)
        fig = plot(samples5)
        plt.savefig('out5/{}.png'.format(str(i5).zfill(3)), bbox_inches='tight')
        i5 += 1
        plt.close(fig)