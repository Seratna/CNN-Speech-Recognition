import numpy as np
import tensorflow as tf
from data_manager import DataManager


class RecognitionNN(object):
    """

    """
    def __init__(self):
        """
        build the graph
        Returns:
            nothing
        """
        IMG_HEIGHT = 129
        IMG_WIDTH = 100
        NUM_CLASSES = 100
        NUM_CHANNELS = 1
        NUM_NEURONS = 1024

        # place holder of input data and label
        x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT*IMG_WIDTH])
        y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

        # reshape each sample (each row of the input) into a image (in the hand written digit case, 20x20)
        x_image = tf.reshape(x, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])  # [#batch, img_height, img_width, #channels]

        # Convolutional Layer 1
        with tf.variable_scope('conv1') as scope:
            # weights shape: [patch_height, patch_width, #input_channels, #output_channels]
            weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1))
            # bias shape: [#output_channel]
            bias = tf.Variable(tf.constant(0.1, shape=[32]))  # 1 bias for each output channel

            conv = tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)

        # max pooling
        pool1 = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolutional Layer 2
        with tf.variable_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[64]))
            conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)

        # max pooling
        pool2 = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolutional Layer 3
        with tf.variable_scope('conv3') as scope:
            weights = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[128]))
            conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)

        # max pooling
        pool3 = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # fully Connected Layer
        with tf.variable_scope('fc1') as scope:
            weights_shape = [int(IMG_HEIGHT/8+1) * int(IMG_WIDTH/8+1) * 128, NUM_NEURONS]  # the img size has been reduced to 1/4
            weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[NUM_NEURONS]))

            pool3_flat = tf.reshape(pool3, [-1, int(IMG_HEIGHT/8+1)*int(IMG_WIDTH/8+1)*128])
            fc1 = tf.nn.relu(tf.matmul(pool3_flat, weights) + bias)

        # Dropout
        keep_prob = tf.placeholder(tf.float32)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        # Readout Layer
        with tf.variable_scope('fc2') as scope:
            weights = tf.Variable(tf.truncated_normal([NUM_NEURONS, NUM_CLASSES], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
            fc2 = tf.nn.softmax(tf.matmul(fc1_drop, weights) + bias)

        # loss function
        loss = -tf.reduce_sum(y * tf.log(fc2))  # cross_entropy

        # training
        training = tf.train.AdamOptimizer(1e-4).minimize(loss)

        # evaluate
        correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
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
        dm = DataManager(100)
        # create a saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print('started new session')

            for i in range(passes):
                # get a batch
                x, y = dm.get_batch(5)
                self.training.run(feed_dict={self.x: x, self.y: y, self.keep_prob: 0.5})
                if i % 10 == 9:
                    train_accuracy = self.accuracy.eval(feed_dict={self.x: x, self.y: y, self.keep_prob: 1.0})
                    print("pass {}, training accuracy {}".format(i+1, train_accuracy))

    @staticmethod
    def get_shuffled_index(size):
        idx = np.array(range(size))
        np.random.shuffle(idx)
        return idx


def main():
    rnn = RecognitionNN()
    rnn.train(10000000)


if __name__ == '__main__':
    main()