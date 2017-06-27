
import tensorflow as tf
import numpy as np
from numpy import loadtxt
import scipy.io as sio
import matplotlib.pyplot as plt
from DeepSNRFunctions import DeepSNRFunctions
from DeepSNRlayers import DeepSNRlayers

sess = tf.InteractiveSession()
np.set_printoptions(threshold=np.nan, precision=3, suppress=True)
tmp = DeepSNRFunctions()
layers = DeepSNRlayers()

lines = tuple(open("CTCF_testSeq.txt", "r"))  # provide input sequences as txt file
mat_contents = sio.loadmat('labels.mat')
labels = mat_contents['labels']

no_samples, len_kernel, len_seq, batch_size = 19600, 35, 100, 100
All_S_Test = tmp.OneHotEncoding(lines, len_kernel, no_samples)

test_input = All_S_Test[:, 140:540]
test_label = labels[39816:59425, :]

####################################################################
# Initialization

x_ = tf.placeholder(tf.float32, shape=[None, 400])
y_ = tf.placeholder(tf.float32, [None, 100])

kernel = loadtxt("W_conv1_2.txt", comments="#", delimiter=",", unpack=False)
W_conv1 = tf.constant(kernel, dtype=tf.float32, shape=[36, 4, 1, 16])

bias0 = loadtxt("thresholds.txt", comments="#", delimiter=",", unpack=False)
b_conv1 = tf.constant(bias0, dtype=tf.float32, shape=[16])

weight1 = loadtxt("W_fc1_2.txt", comments="#", delimiter=",", unpack=False)
W_fc1 = tf.constant(weight1, dtype=tf.float32, shape=[1*1*16, 1])

b_fc1 = -5.6241183

b_fc1_d = -0.00069609686

weight1_d = loadtxt("W_fc1_d_2.txt", comments="#", delimiter=",", unpack=False)
W_fc1_d = tf.constant(weight1_d, dtype=tf.float32, shape=[1, 16])

bias0_d = loadtxt("b_conv1_d_2.txt", comments="#", delimiter=",", unpack=False)
b_conv1_d = tf.constant(bias0_d, dtype=tf.float32, shape=[16])

kernel_d = loadtxt("W_conv1_d_2.txt", comments="#", delimiter=",", unpack=False)
W_conv1_d = tf.constant(kernel_d, dtype=tf.float32, shape=[36, 4, 1, 16])

th = 0.55

sess.run(tf.initialize_all_variables())

#################################################################
# Convolution

x_image = tf.reshape(x_, [-1, 100, 4, 1])

h_conv1 = layers.conv2d(x_image, W_conv1)

mean, variance = tf.nn.moments(h_conv1, [0, 1, 2])

h_conv1_BN = tf.nn.relu(tf.nn.batch_normalization(h_conv1, mean, variance, 0, 1, 1e-5))

h_pool1, switch = layers.max_pooling(h_conv1_BN, len_seq, len_kernel)

h_pool1_flat = tf.reshape(h_pool1, [-1, 1*1*16])

h_fc1 = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

#################################################################
# Deconvolution

h_pool1_d = tf.matmul(h_fc1, W_fc1_d) + b_fc1_d

h_pool1_d_flat = tf.reshape(h_pool1, [-1, 1, 1, 16])

h_conv1_d = layers.unpool_layer2x2_batch(h_pool1_d_flat, switch, [batch_size, (len_seq-len_kernel), 1, 16])

S_d = layers.deconv(h_conv1_d, W_conv1_d, b_conv1_d, batch_size, len_seq, len_kernel)

mean_d, variance_d = tf.nn.moments(S_d, [0, 1, 2])

S_d_BN = tf.nn.relu(tf.nn.batch_normalization(S_d, mean_d, variance_d, 0, 1, 1e-5))

S_d_max = tf.nn.max_pool(S_d_BN, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='VALID')

y = tf.reshape(S_d_max, [-1, 100])

y_score = tf.nn.sigmoid(y)

#################################################################
# Testing and Results

y_binary = tf.greater(y_score, th)

y_out = np.zeros((no_samples, len_seq), dtype=np.float)

for j in range(no_samples/batch_size):
    randIndx_test = np.arange(j*batch_size, (j+1)*batch_size)
    test_xs = test_input[randIndx_test, :]
    test_ys = test_label[randIndx_test, :]
    y_out_ = sess.run(y_binary, feed_dict={x_: test_xs, y_: test_ys})
    y_out[j*batch_size:(j+1)*batch_size, :] = y_out_

# np.savetxt('y_binary_out.out', y_out, delimiter=',')  # uncomment to save prediction output
eval_metric = tmp.PerformanceEval(y_out, test_label, no_samples)

plt.boxplot(eval_metric)
plt.xticks([1, 2, 3, 4], ['Precision', 'Recall', 'F1-score', 'IoU'])
plt.show()
