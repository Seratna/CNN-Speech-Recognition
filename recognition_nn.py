import numpy as np
import tensorflow as tf
from digits_manager import DataManager
import sys
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import scipy.stats.mstats


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
        IMG_WIDTH = None
        NUM_CLASSES = 11
        NUM_CHANNELS = 1
        NUM_NEURONS = 1024

        # place holder of input data and label
        x = tf.placeholder(tf.float32, shape=[IMG_HEIGHT, None])
        y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

        # reshape each sample (each row of the input) into a image (in the hand written digit case, 20x20)
        x_image = tf.reshape(x, [1, IMG_HEIGHT, -1, NUM_CHANNELS])  # [#batch, img_height, img_width, #channels]

        # Convolutional Layer 1
        with tf.variable_scope('conv1') as scope:
            # weights shape: [patch_height, patch_width, #input_channels, #output_channels]
            weights = tf.Variable(tf.truncated_normal([10, 10, NUM_CHANNELS, 32], stddev=0.1))
            # bias shape: [#output_channel]
            bias = tf.Variable(tf.constant(0.1, shape=[32]))  # 1 bias for each output channel

            conv = tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)

        # max pooling
        pool1 = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding='SAME')

        # Convolutional Layer 2
        with tf.variable_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal([10, 10, 32, 64], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[64]))
            conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(conv + bias)

        # max pooling
        pool2 = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding='SAME')

        # fully Connected Layer
        with tf.variable_scope('fc1') as scope:
            weights_shape = [int(IMG_HEIGHT/4+1), 1, 64, NUM_NEURONS]  # the img size has been reduced to 1/4
            weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[NUM_NEURONS]))

            conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='VALID')
            fc1 = tf.nn.relu(conv + bias)

        # Dropout
        keep_prob = tf.placeholder(tf.float32)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        # Readout Layer
        with tf.variable_scope('fc2') as scope:
            weights = tf.Variable(tf.truncated_normal([NUM_NEURONS, NUM_CLASSES], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))

            fc1_flat = tf.reshape(fc1_drop, [-1, NUM_NEURONS])
            fc2 = tf.nn.softmax(tf.matmul(fc1_flat, weights) + bias)

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

    def train(self, passes, resume_training=False):
        dm = DataManager()
        saver = tf.train.Saver()  # create a saver

        global_step = 0
        with tf.Session() as sess:
            if resume_training:  # restore from latest check point
                with open('saver/checkpoint') as file:
                    line = file.readline()
                    ckpt = line.split('"')[1]
                    global_step = int(ckpt.split('-')[1])
                # restore
                saver.restore(sess, 'saver/'+ckpt)
                print('restored from checkpoint ' + ckpt)
            else:
                sess.run(tf.initialize_all_variables())
                print('started new session')

            for step in range(1+global_step, passes+global_step):
                # get a batch
                x, y = dm.get_image('training')
                self.training.run(feed_dict={self.x: x, self.y: y.T, self.keep_prob: 0.5})

                if step % 10 == 0:
                    train_accuracy = self.accuracy.eval(feed_dict={self.x: x, self.y: y.T, self.keep_prob: 1.0})
                    print("pass {}, training accuracy {}".format(step, train_accuracy))

                if step % 1000 == 0:  # save weights
                    saver.save(sess, 'saver/cnn', global_step=step)
                    print('checkpoint saved')

    def test(self):
        dm = DataManager()
        saver = tf.train.Saver()  # create a saver

        with tf.Session() as sess:
            ckpt = 'saver/cnn-5x5-18000'
            saver.restore(sess, ckpt)
            print('restored from checkpoint ' + ckpt)

            x, y = dm.get_image('test')
            fc2 = self.fc2.eval(feed_dict={self.x: x, self.keep_prob: 1.0}).T

            # correct = np.argmax(fc2, 0) == np.argmax(y, 0)
            # print(np.mean(correct))

            print(self.output_to_words(y))
            print(self.output_to_words(fc2))

            # plt.imshow(fc2)
            # plt.show()

    def output_to_words(self, fc2):
        prediction = np.argmax(fc2, 0)

        l = []
        window_size = 31
        for i in range(prediction.shape[0]-window_size+1):
            l.append(int(scipy.stats.mstats.mode(prediction[i:i+window_size])[0]))

        pre = None
        labels = []
        for i in l:
            if i != pre:
                labels.append(i)
                pre = i

        return labels


def main():
    passes = int(sys.argv[1])
    resume_training = sys.argv[2].lower() in ['true', '1', 'y', 'yes']

    rnn = RecognitionNN()
    # rnn.train(passes, resume_training)
    rnn.test()


if __name__ == '__main__':
    main()
