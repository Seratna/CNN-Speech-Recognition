import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import tensorflow as tf


class ToyCNN(object):
    """
    this is a very simple example of using CNN (Convolutional Neural Network)
    to recognize hand written digits.
    """
    def __init__(self):
        """
        build the graph
        Returns:
            nothing
        """
        IMG_SIZE = 20  # each image of a digit is 20x20
        NUM_CLASSES = 10  # number 0~9
        NUM_CHANNELS = 1  # grey scale (only 1 channel)
        NUM_NEURONS = 1024  # number of neurons in the fully connected layer

        # place holder of input data and label
        x = tf.placeholder(tf.float32, shape=[None, 20*40])
        y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

        # reshape each sample (each row of the input) into a image (in the hand written digit case, 20x20)
        x_image = tf.reshape(x, [-1, IMG_SIZE, 40, NUM_CHANNELS])  # [#batch, img_height, img_width, #channels]

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
            weights = tf.Variable(tf.truncated_normal([5, 5, 32, 32], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[32]))
            conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)

        # max pooling
        pool2 = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # fully Connected Layer
        with tf.variable_scope('fc1') as scope:
            weights_shape = [int(IMG_SIZE/4)*int(IMG_SIZE/4) * 32, NUM_NEURONS]  # the img size has been reduced to 1/4
            weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[NUM_NEURONS]))

            pool2_flat = tf.reshape(pool2, [-1, int(IMG_SIZE/4)*int(IMG_SIZE/4)*32])
            fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + bias)

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

    def train(self, xx, yy, batch_size, passes):
        # create a saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print('started new session')

            for i in range(passes):
                # get a batch
                r, c = xx.shape
                batch = self.get_shuffled_index(r)[:batch_size]
                x = xx[batch]
                y = yy[batch]
                self.training.run(feed_dict={self.x: x, self.y: y, self.keep_prob: 0.5})
                if i % 10 == 9:
                    train_accuracy = self.accuracy.eval(feed_dict={self.x: x, self.y: y, self.keep_prob: 1.0})
                    print("pass {}, training accuracy {}".format(i+1, train_accuracy))

            # saver weights
            saver.save(sess, 'saver/cnn', global_step=passes)

    def predict(self, xx, weights_file):
        # create a saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, weights_file)
            fc = self.fc2.eval(feed_dict={self.x: xx, self.keep_prob: 1.0})
            print(fc)

    @staticmethod
    def get_shuffled_index(size):
        idx = np.array(range(size))
        np.random.shuffle(idx)
        return idx


def main():
    # load data from HandWrittenDigit.mat.
    # This data file is adopt from Andrew Ng's Coursera Machine learning course homework
    mat = io.loadmat('HandWrittenDigits')
    digits = mat["X"]  # x is a 5000x400 matrix, each row is a training sample
    labels_raw = mat["y"]  # y is a 5000x1 matrix, contains the corresponding labels to each training sample
    labels = labels_raw.dot(np.ones((1, 10))) == np.ones((5000, 10)).cumsum(1)  # convert the label to 0-1 label

    # random shuffle the samples and split them into training set and testing set
    idx = np.array(range(5000))
    np.random.shuffle(idx)
    x = digits[idx[:4000]]  # training set
    y = labels[idx[:4000]]
    xx = digits[idx[4000:]]  # testing set
    yy = labels[idx[4000:]]

    # TODO visualize training set

    # training
    toy = ToyCNN()
    toy.train(x, y, 100, 500)

    # # predict
    # for i in toy.get_shuffled_index(1000):
    #     print(i)
    #     digit = xx[i].reshape((20, 20)).T
    #     sample = np.zeros((20, 40))
    #     sample[:, 10:30] = digit
    #     plt.imshow(sample)
    #     plt.show()
    #     toy.predict(sample.reshape((-1, 800)), 'saver/cnn-500')


if __name__ == '__main__':
    main()