import tensorflow as tf
import tensorflow.contrib.slim as slim


class StudentNetwork(object):
    def __init__(self, num_of_class):
        self.num_of_class = num_of_class

    def build_network(self, in_batch, reuse, is_train, dropout_keep_prob=.8):
        dropout_keep_prob = 1 if not is_train else dropout_keep_prob
        with tf.variable_scope('SqueezeNeXt', reuse=reuse):
            bn_param = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_train,
                'activation_fn': tf.nn.relu
            }
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=bn_param,
                                biases_initializer=None, weights_regularizer=slim.l2_regularizer(0.0005),
                                activation_fn=tf.identity):
                endpoint = slim.conv2d(in_batch, 64, 5, 2, 'VALID', scope='prev_conv')
                endpoint = tf.nn.max_pool(endpoint, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

                with tf.variable_scope('Util_1'):
                    endpoint = slim.conv2d(endpoint, 64, 1, 1)
                    for i in range(2):
                        endpoint = self.squeeze_next_block(endpoint, 64, i)

                with tf.variable_scope('Util_2'):
                    endpoint = slim.conv2d(endpoint, 128, 1, 2)
                    for i in range(4):
                        endpoint = self.squeeze_next_block(endpoint, 128, i)

                with tf.variable_scope('Util_3'):
                    endpoint = slim.conv2d(endpoint, 256, 1, 2)
                    for i in range(14):
                        endpoint = self.squeeze_next_block(endpoint, 256, i)

                with tf.variable_scope('Util_4'):
                    endpoint = slim.conv2d(endpoint, 512, 1, 2)
                    endpoint = self.squeeze_next_block(endpoint, 512, 0)

                endpoint = slim.conv2d(endpoint, 128, 1, 1, scope='post_conv')
                endpoint = tf.reduce_mean(endpoint, [1, 2], keepdims=False)
                endpoint = slim.dropout(endpoint, dropout_keep_prob, is_training=is_train, scope='Dropout')

                prev_logit = endpoint
                endpoint = slim.fully_connected(endpoint, self.num_of_class, activation_fn=tf.identity)
        return endpoint, prev_logit

    @staticmethod
    def squeeze_next_block(in_batch, num_output, scope_num):
        with tf.variable_scope('SqueezeNeXt_block{:d}'.format(scope_num)):
            skip_branch = in_batch

            endpoint = slim.conv2d(in_batch, int(num_output / 2), 1, 1, scope='conv_1')
            endpoint = slim.conv2d(endpoint, int(num_output / 4), 1, 1, scope='conv_2')
            endpoint = slim.conv2d(endpoint, int(num_output / 2), [3, 1], 1, scope='conv_3')
            endpoint = slim.conv2d(endpoint, int(num_output / 2), [1, 3], 1, scope='conv_4')
            endpoint = slim.conv2d(endpoint, num_output, 1, 1, scope='conv_5')

            return skip_branch + endpoint





