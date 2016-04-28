import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy import misc


class DataManager(object):
    def __init__(self, dict_size):
        """

        Args:
            dict_size: number of words to recognize

        Returns:

        """
        # load data index
        with open('TIMIT/train_index.txt') as file:
            self.training_samples = [path.replace('\n', '') for path in file]
        with open('TIMIT/test_index.txt') as file:
            self.testing_samples = [path.replace('\n', '') for path in file]

        # build dictionary
        self.dictionary = self.build_dictionary(dict_size)

    def get_paths(self, size, data_set=0):
        """

        Args:
            size: number of images to look at
            data_set: 0 for training data, other for testing data

        Returns:

        """
        if data_set == 0:  # training
            samples = self.training_samples
        else:  # testing
            samples = self.testing_samples

        index = np.array(range(len(samples)), dtype=np.int32)
        np.random.shuffle(index)

        paths = [samples[i] for i in index[:size]]

        return paths

    def get_batch(self, size):
        """
        get a batch of data for recognition training

        Args:
            size: number of images to look at
            data_set: 0 for training data, other for testing data

        Returns:
            images: each row is a flattened image of a word
            keys: each row is a vector which has the size of dictionary

        """
        paths = self.get_paths(size, data_set=0)
        images = []
        keys = []

        for path in paths:
            im = mpimg.imread(path + '.png')
            words, pixels = self.get_words(path)
            for start, end, word in zip(pixels[:-1], pixels[1:], words):
                if word not in self.dictionary:  # if the word is not in the dictionary, then skip it
                    continue

                # get the segmented image of a word, and resize it if necessary
                segment = im[:, start:end]
                if (end-start) <= 100:
                    image = np.zeros((129, 100))
                    image[:, :end-start] = segment
                else:
                    image = misc.imresize(segment, (129, 100))
                images.append(image.flatten())

                # build word vector
                key = np.zeros(len(self.dictionary))
                key[self.dictionary[word]] = 1
                keys.append(key)

        return np.array(images), np.array(keys)

    @staticmethod
    def get_words(path):
        """
        get the segmentation points for a given image.

        Args:
            path:

        Returns:

        """
        # read the start and end points from the file
        words = []
        points = []
        with open(path + '.WRD') as file:
            for line in file:
                elements = line.replace('\n', '').split(' ')
                points.append((int(elements[0]), int(elements[1])))
                words.append(elements[2])

        # put the segment points in a array
        segment = np.array([p[0] for p in points] + [points[-1][1]], dtype=np.int32)
        pixels = np.array(segment/128, dtype=np.int32)

        assert len(pixels)-len(words) == 1

        return words, pixels

    def get_segmentation_sample(self, data_set=0):
        path = self.get_paths(1, data_set)[0]

        # read the corresponding image
        im = mpimg.imread(path + '.png')
        r, c = im.shape  # get the size of the image

        key = np.zeros(c, dtype=np.int32)  # make a zero vector
        words, pixels = self.get_words(path)
        key[pixels] = 1  # mark the corresponding segmentation pixels to 1

        return im, key

    def build_dictionary(self, dict_size):
        """
        build a dict of word-index pairs. Words are sorted according to the frequency in the training data set.
        Index here is the index of the bag-of-words vector.

        Args:
            dict_size: number of words in the bag

        Returns:

        """
        dictionary = {}
        for path in self.training_samples:
            with open(path + '.WRD') as file:
                for line in file:
                    word = line.replace('\n', '').split(' ')[2]
                    dictionary[word] = dictionary[word]+1 if word in dictionary else 1
        words = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

        # build word-index pairs
        index = {}
        for i in range(dict_size):
            index[words[i][0]] = i

        return index


def main():
    dm = DataManager(100)
    dm.get_batch(5)
    # im, key = dm.get_segmentation_sample()


if __name__ == '__main__':
    main()
