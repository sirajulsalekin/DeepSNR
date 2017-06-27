import tensorflow as tf


class DeepSNRlayers(object):

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')

    def max_pooling(self, x, len_seq, len_kernel):
        max_value = tf.nn.max_pool(x, ksize=[1, len_seq-len_kernel, 1, 1], strides=[1, len_seq-len_kernel, 1, 1], padding='VALID')
        _, max_id = tf.nn.max_pool_with_argmax(x, ksize=[1, len_seq-len_kernel, 1, 1], strides=[1, len_seq-len_kernel, 1, 1], padding='VALID')
        return max_value, max_id

    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.pack(output_list)

    def unpool_layer2x2_batch(self, bottom, argmax, top_shape):
        bottom_shape = tf.shape(bottom)

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = self.unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size*(width//1)*(height//height)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height//height, width//1, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels*(width//1)*(height//height)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height//height, width//1, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat(4, [t2, t3, t1])
        indices = tf.reshape(t, [(height//height)*(width//1)*channels*batch_size, 4])

        x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
        return tf.sparse_add(tf.zeros(tf.to_int32(delta.shape)), tf.sparse_reorder(delta))

    def deconv(self, value, w, bias, batch_size, len_seq, len_kernel):
        c_d = tf.to_float(tf.reshape(value, [batch_size, len_seq-len_kernel, 1, 16])) + bias
        deconv_shape = tf.pack([batch_size, len_seq, 4, 1])
        deconv_out = tf.nn.conv2d_transpose(c_d, w, output_shape=deconv_shape, strides=[1, 1, 1, 1], padding='VALID')
        return deconv_out

