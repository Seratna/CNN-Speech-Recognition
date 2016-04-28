import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


class DataManager(object):
    def __init__(self):
        # load data index
        with open('TIMIT/train_index.txt') as file:
            self.training_samples = [path.replace('\n', '') for path in file]
        with open('TIMIT/test_index.txt') as file:
            self.testing_samples = [path.replace('\n', '') for path in file]

    def get_paths(self, size, data_set = 0):
        """

        :param size:
        :param data_set:
        :return:
        """
        if data_set == 0:  # training
            samples = self.training_samples
        else:  # testing
            samples = self.testing_samples

        index = np.array(range(len(samples)), dtype=np.int32)
        np.random.shuffle(index)

        paths = [samples[i] for i in index[:size]]

        return paths

    def get_batch(self, size, data_set = 0):
        """

        :param data_set: 0 is training data, 1 is test data
        :return:
        """
        paths = self.get_paths(size, data_set)
        l = []
        for path in paths:
            points = self.get_segment_points(path)
            print(points)

    def get_segment_points(self, path):
        """

        :param path:
        :return:
        """
        # read the start and end points from the file
        with open(path + '.WRD') as file:
            points = []
            for line in file:
                elements = line.split(' ')
                points.append((int(elements[0]), int(elements[1])))

        # put the segment points in a array
        segment = np.array([p[0] for p in points] + [points[-1][1]], dtype=np.int32)
        return segment

    def get_segmentation_sample(self, data_set = 0):
        path = self.get_paths(1, data_set)[0]

        segment = self.get_segment_points(path)

        # read the corresponding image
        im = mpimg.imread(path + '.png')
        r, c = im.shape  # get the size of the image

        key = np.zeros(c, dtype=np.int32)  # make a zero vector
        pixels = np.array(segment/128, dtype=np.int32)
        key[pixels] = 1  # mark the corresponding segmentation pixels to 1

        # img = im.copy()
        # img[:, pixels] = 0
        # plt.imshow(img)
        # plt.show()
        # print(key)

        return im, key


def main():
    dm = DataManager()
    dm.get_batch(5)
    im, key = dm.get_segmentation_sample()

    plt.imshow(im)
    plt.show()
    print(key)


    # a = np.array([[1,2,3], [4,5,6]])
    # b = np.array([[7,8,9], [10,11,12]])
    # print(a)
    # print(b)
    # c = np.array([a, b])
    # print(c)
    # print(c.shape)


if __name__ == '__main__':
    main()