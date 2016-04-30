import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


class DataManager(object):
    def __init__(self):
        pass

    def get_image(self, flag):
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'O', 'Z']
        index = {}
        for i in range(11):
            index[numbers[i]] = i
        np.random.shuffle(numbers)

        images = []
        keys = []

        if flag == 'training':
            start = 1
            end = 51
        else:
            start = 51
            end = 75

        for number in numbers:
            im_n = np.random.randint(start, end)
            file_name = 'DIGITS/img/{}_{}.png'.format(number, im_n)
            im = mpimg.imread(file_name)
            images.append(im)

            r, c = im.shape
            key = np.zeros((11, c))
            key[index[number], :] = 1
            keys.append(key)

        image = np.concatenate(images, axis=1)
        key = np.concatenate(keys, axis=1)

        return image, key


def copy_raw_file():
    import os
    import shutil

    org = '/Users/Antares/Downloads/isolated_digits_ti_train_endpt/MAN'
    dst = '/Users/Antares/Downloads/digits/raw'
    ds = [d for d in os.listdir(org) if d[0] != '.']

    counter = {}
    for d in ds:
        fs = [f for f in os.listdir(org + '/' + d) if f[0] != '.']
        for f in fs:
            file_name = '{}/{}/{}'.format(org, d, f)
            content = f[0]

            if content in counter:
                counter[content] += 1
            else:
                counter[content] = 1

            target = '{}/{}_{}.wav'.format(dst, content, counter[content])
            print(target)

            shutil.copyfile(file_name, target)



def main():
    dm = DataManager()
    x, y = dm.get_image('training')
    plt.imshow(x)
    plt.show()


if __name__ == '__main__':
    main()