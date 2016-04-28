import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from data_manager import DataManager


class SegmentationNN(object):
    """

    """
    def __init__(self):
        """
        build the graph
        Returns:
            nothing
        """
        IMG_HEIGHT = 129  #
        INPUT_CHANNELS = 1  #
        NUM_NEURONS = 1024  # number of neurons in the fully connected layer

        # place holder of input data and label
        x = tf.placeholder(tf.float32, shape=[IMG_HEIGHT, None])
        y = tf.placeholder(tf.float32, shape=[None])

        # reshape each sample (each row of the input) into a image (in the hand written digit case, 20x20)
        x_image = tf.reshape(x, [1, IMG_HEIGHT, -1, INPUT_CHANNELS])  # [#batch, img_height, img_width, #channels]

        # Convolutional Layer 1
        with tf.variable_scope('conv1') as scope:
            # weights shape: [patch_height, patch_width, #input_channels, #output_channels]
            weights = tf.Variable(tf.truncated_normal([10, 10, INPUT_CHANNELS, 32], stddev=0.1))
            # bias shape: [#output_channel]
            bias = tf.Variable(tf.constant(0.1, shape=[32]))  # 1 bias for each output channel

            conv = tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)

        # max pooling
        pool1 = tf.nn.max_pool(relu, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        # Convolutional Layer 2
        with tf.variable_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal([10, 10, 32, 64], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[64]))
            conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)

        # max pooling
        pool2 = tf.nn.max_pool(relu, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        # Convolutional Layer 3
        with tf.variable_scope('conv3') as scope:
            weights = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[64]))
            conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)

        # max pooling
        pool3 = tf.nn.max_pool(relu, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        # fully Connected Layer
        with tf.variable_scope('fc1') as scope:
            weights_shape = [int(IMG_HEIGHT/8+1), 1, 64, NUM_NEURONS]
            weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[NUM_NEURONS]))
            conv = tf.nn.conv2d(pool3, weights, strides=[1, 1, 1, 1], padding='VALID')
            fc1 = tf.nn.relu(conv + bias)

        # Dropout
        keep_prob = tf.placeholder(tf.float32)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        # Readout Layer
        with tf.variable_scope('fc2') as scope:
            weights = tf.Variable(tf.truncated_normal([1, 1, NUM_NEURONS, 1], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[1]))
            conv = tf.nn.conv2d(fc1_drop, weights, strides=[1, 1, 1, 1], padding='SAME')
            fc2 = tf.reshape(conv + bias, [-1])

        # loss function
        loss = -tf.reduce_sum(y * tf.log(fc2))  # cross_entropy

        # training
        training = tf.train.AdamOptimizer(1e-4).minimize(loss)

        # evaluate
        correct_prediction = tf.equal(tf.cast(tf.greater(fc2, 0), dtype=tf.int32),
                                      tf.cast(tf.greater(y, 0), dtype=tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #
        self.x = x
        self.y = y
        self.keep_prob = keep_prob

        self.fc2 = fc2
        self.loss = loss
        self.training = training
        self.accuracy = accuracy

    def train(self, passes):
        dm = DataManager()
        # create a saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print('started new session')

            for i in range(passes):
                print("pass {}".format(i+1))
                # get a batch
                x, y = dm.get_batch(1)
                self.training.run(feed_dict={self.x: x, self.y: y, self.keep_prob: 0.5})
                if i % 10 == 9:
                    train_accuracy = self.accuracy.eval(feed_dict={self.x: x, self.y: y, self.keep_prob: 1.0})
                    print("pass {}, training accuracy {}".format(i+1, train_accuracy))

                if i % 1000 == 9:
                    # saver weights
                    saver.save(sess, 'saver/snn', global_step=passes)

    # def predict(self, xx, weights_file):
    #     # create a saver
    #     saver = tf.train.Saver()
    #
    #     with tf.Session() as sess:
    #         saver.restore(sess, weights_file)
    #         fc = self.fc2.eval(feed_dict={self.x: xx, self.keep_prob: 1.0})
    #         print(fc)


def main():
    snn = SegmentationNN()
    snn.train(100000)


if __name__ == '__main__':
    main()