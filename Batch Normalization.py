import numpy as np
import tensorflow as tf , tqdm
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("data/", one_hot = True)
batch_size = 60
total_batch = int(mnist.train.num_examples/batch_size)
learning_rate = 20  # 10 , 0.05 , 0.01 ,20
training_epochs = 2
mu, sigma = 0, 0.2 #0.1 , 1 , 10, 20

w1_initial = np.random.normal(mu,sigma, size=(28*28,100)).astype(np.float32)
w2_initial = np.random.normal(mu,sigma,size=(100,100)).astype(np.float32)
w3_initial = np.random.normal(mu,sigma,size=(100,100)).astype(np.float32)
w4_initial = np.random.normal(mu,sigma,size=(100,10)).astype(np.float32)

epsilon = 1e-3

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# Layer 1 without BN
w1 = tf.Variable(w1_initial)
b1 = tf.Variable(tf.zeros([100]))
z1 = tf.matmul(x,w1)+b1
l1 = tf.nn.sigmoid(z1)

w1_BN = tf.Variable(w1_initial)
z1_BN = tf.matmul(x,w1_BN)
batch_mean1, batch_var1 = tf.nn.moments(z1_BN,[0])
z1_hat = (z1_BN - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

scale1 = tf.Variable(tf.ones([100]))
beta1 = tf.Variable(tf.zeros([100]))

BN1 = scale1 * z1_hat + beta1
l1_BN = tf.nn.sigmoid(BN1)

w2 = tf.Variable(w2_initial)
b2 = tf.Variable(tf.zeros([100]))
z2 = tf.matmul(l1,w2)+b2
l2 = tf.nn.sigmoid(z2)

w2_BN = tf.Variable(w2_initial)
z2_BN = tf.matmul(l1_BN,w2_BN)
batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
scale2 = tf.Variable(tf.ones([100]))
beta2 = tf.Variable(tf.zeros([100]))
BN2 = tf.nn.batch_normalization(z2_BN,batch_mean2,batch_var2,beta2,scale2,epsilon)
l2_BN = tf.nn.sigmoid(BN2)

w3 = tf.Variable(w3_initial)
b3 = tf.Variable(tf.zeros([100]))
z3 = tf.matmul(l2,w3)+b3
l3 = tf.nn.sigmoid(z3)

w3_BN = tf.Variable(w3_initial)
z3_BN = tf.matmul(l2_BN,w3_BN)
batch_mean3, batch_var3 = tf.nn.moments(z3_BN,[0])
scale3 = tf.Variable(tf.ones([100]))
beta3 = tf.Variable(tf.zeros([100]))
BN3 = tf.nn.batch_normalization(z3_BN,batch_mean3,batch_var3,beta3,scale3,epsilon)
l3_BN = tf.nn.sigmoid(BN3)

w4 = tf.Variable(w4_initial)
b4 = tf.Variable(tf.zeros([10]))
y  = tf.nn.softmax(tf.matmul(l3,w4)+b4)

w4_BN = tf.Variable(w4_initial)
b4_BN = tf.Variable(tf.zeros([10]))
z4_BN = tf.matmul(l3_BN,w4_BN)+b4_BN
batch_mean4, batch_var4 = tf.nn.moments(z4_BN,[0])
scale4 = tf.Variable(tf.ones([10]))
beta4 = tf.Variable(tf.zeros([10]))
BN4 = tf.nn.batch_normalization(z4_BN,batch_mean4,batch_var4,beta4,scale4,epsilon)
y_BN  = tf.nn.softmax(BN4)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy_BN = -tf.reduce_sum(y_*tf.log(y_BN))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step_BN = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_BN)

correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
correct_prediction_BN = tf.equal(tf.arg_max(y_BN,1),tf.arg_max(y_,1))
accuracy_BN = tf.reduce_mean(tf.cast(correct_prediction_BN,tf.float32))


zs, BNs, acc,loss,loss_BN, acc_BN = [] , [] , [] , [] , [] , []

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(training_epochs):
    epoch_accr1 = 0
    epoch_accr2 = 0
    for i in range(total_batch):
        batch = mnist.train.next_batch(batch_size)
        batch_accr1 = train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        batch_accr2 = train_step_BN.run(feed_dict={x: batch[0], y_: batch[1]})
        #epoch_accr1 += batch_accr1 / total_batch
        #epoch_accr2 += batch_accr2 / total_batch
        if i % 20 is 0:
            res = sess.run([accuracy,accuracy_BN,z3,BN3],feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            acc.append(res[0])
            acc_BN.append(res[1])
            print('Acuuracy:',res[0] , 'BN_Accuracy:', res[1])
            #loss.append(l[0])
            #loss_BN.append(l[1])
            zs.append(np.mean(res[2],axis=0)) # record the mean value of z3 over the entire test set
            BNs.append(np.mean(res[3],axis=0)) # record the mean value of BN3 over the entire test set

zs, BNs, acc, acc_BN ,loss,loss_BN = np.array(zs), np.array(BNs), np.array(acc), np.array(acc_BN),np.array(loss),np.array(loss_BN)
fig, ax = plt.subplots()

ax.plot(range(0,len(acc)*50,50),acc,label='Without BN')
ax.plot(range(0,len(acc)*50,50),acc_BN, label='With BN')

ax.set_xlabel('Training steps')
ax.set_ylabel('Loss')
ax.set_ylim([0,1])
ax.set_title('Accuracy')
ax.legend(loc=4)
plt.show()