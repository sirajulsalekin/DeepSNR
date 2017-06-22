
import tensorflow as tf
import numpy as np
from numpy import loadtxt
import scipy.io as sio
sess = tf.InteractiveSession()
np.set_printoptions(threshold=np.nan, precision=3, suppress=True)

lines = tuple(open("CTCF_testSeq.txt", "r"))
A = np.array([[1.0, 0, 0, 0]])
C = np.array([[0, 1.0, 0, 0]])
G = np.array([[0, 0, 1.0, 0]])
T = np.array([[0, 0, 0, 1.0]])

no_samples, len_kernel, len_seq, batch_size = 19600, 35, 100, 100

S_ = 0.25 * np.ones((no_samples, 400+len_kernel*8), dtype=np.float32 )
S_[:, len_kernel*4:len_kernel*4+400] = np.zeros((no_samples, 400), dtype=np.float32)

for k in range(0, no_samples):
    in_seq = lines[k]

    for j in range(0, 100):
        if in_seq[j] == 'A':
            S_[k, (len_kernel+j)*4:(len_kernel+j+1)*4] = A
        elif in_seq[j] == 'C':
            S_[k, (len_kernel+j)*4:(len_kernel+j+1)*4] = C
        elif in_seq[j] == 'G':
            S_[k, (len_kernel+j)*4:(len_kernel+j+1)*4] = G
        else:
            S_[k, (len_kernel+j)*4:(len_kernel+j+1)*4] = T

mat_contents = sio.loadmat('labels.mat')
labels = mat_contents['labels']
All_S_Test = S_

test_input = All_S_Test[:, 140:540]
test_label = labels[39816:59425, :]


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


def max_pooling(x):
    max_value = tf.nn.max_pool(x, ksize=[1, len_seq-len_kernel, 1, 1], strides=[1, len_seq-len_kernel, 1, 1], padding='VALID')
    _, max_id = tf.nn.max_pool_with_argmax(x, ksize=[1, len_seq-len_kernel, 1, 1], strides=[1, len_seq-len_kernel, 1, 1], padding='VALID')
    return max_value, max_id


def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.pack(output_list)


def unpool_layer2x2_batch(bottom, argmax, top_shape):
    bottom_shape = tf.shape(bottom)

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*(width//1)*(height//(len_seq-len_kernel))])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//(len_seq-len_kernel), width//1, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*(width//1)*(height//(len_seq-len_kernel))])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height//(len_seq-len_kernel), width//1, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat(4, [t2, t3, t1])
    indices = tf.reshape(t, [(height//(len_seq-len_kernel))*(width//1)*channels*batch_size, 4])

    x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    return tf.sparse_add(tf.zeros(tf.to_int32(delta.shape)), tf.sparse_reorder(delta))


def deconv(value, w, bias):
    c_d = tf.to_float(tf.reshape(value, [batch_size, len_seq-len_kernel, 1, 16])) + bias
    deconv_shape = tf.pack([batch_size, len_seq, 4, 1])
    deconv_out = tf.nn.conv2d_transpose(c_d, w, output_shape=deconv_shape, strides=[1, 1, 1, 1], padding='VALID')
    return deconv_out

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

h_conv1 = conv2d(x_image, W_conv1)

mean, variance = tf.nn.moments(h_conv1, [0, 1, 2])

h_conv1_BN = tf.nn.relu(tf.nn.batch_normalization(h_conv1, mean, variance, 0, 1, 1e-5))

h_pool1, switch = max_pooling(h_conv1_BN)

h_pool1_flat = tf.reshape(h_pool1, [-1, 1*1*16])

h_fc1 = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

#################################################################
# Deconvolution

h_pool1_d = tf.matmul(h_fc1, W_fc1_d) + b_fc1_d

h_pool1_d_flat = tf.reshape(h_pool1, [-1, 1, 1, 16])

h_conv1_d = unpool_layer2x2_batch(h_pool1_d_flat, switch, [batch_size, (len_seq-len_kernel), 1, 16])

S_d = deconv(h_conv1_d, W_conv1_d, b_conv1_d)

mean_d, variance_d = tf.nn.moments(S_d, [0, 1, 2])

S_d_BN = tf.nn.relu(tf.nn.batch_normalization(S_d, mean_d, variance_d, 0, 1, 1e-5))

S_d_max = tf.nn.max_pool(S_d_BN, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='VALID')

y = tf.reshape(S_d_max, [-1, 100])

y_score = tf.nn.sigmoid(y)

#################################################################
# Testing

correct_prediction = tf.equal(tf.to_float(tf.greater(y_score, th)), y_)

intersection = tf.logical_and(tf.greater(y_score, th), tf.equal(y_, 1))

union = tf.logical_or(tf.greater(y_score, th), tf.equal(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

recall = tf.div(tf.reduce_sum(tf.cast(intersection, tf.float32)), tf.reduce_sum(tf.cast(tf.equal(y_, 1), tf.float32)))

precision = tf.div(tf.reduce_sum(tf.cast(intersection, tf.float32)), tf.reduce_sum(tf.cast(tf.greater(y_score, th), tf.float32)))

F1 = 2*tf.div(tf.multiply(recall, precision), tf.add(recall, precision))

IoU = tf.div(tf.reduce_sum(tf.cast(intersection, tf.float32)), tf.reduce_sum(tf.cast(union, tf.float32)))

validation = np.zeros((196, 4), dtype=np.float)

y_out = np.zeros((19600, 100), dtype=np.float)

for j in range(196):  # 16
    randIndx_test = np.arange(j*batch_size, (j+1)*batch_size)
    test_xs = test_input[randIndx_test, :]
    test_ys = test_label[randIndx_test, :]
    F1_, recall_, precision_, IoU_ = sess.run([F1, recall, precision, IoU], feed_dict={x_: test_xs, y_: test_ys})
    validation[j, 0] = precision_
    validation[j, 1] = recall_
    validation[j, 2] = F1_
    validation[j, 3] = IoU_

valid_mean = np.median(validation, axis=0)

print("Threshold %g, Precision %g, Recall %g, F1 %g, IoU %g" % (th, valid_mean[0], valid_mean[1], valid_mean[2], valid_mean[3]))
