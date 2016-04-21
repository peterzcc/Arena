import numpy
from arena.utils import *
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
import logging


'''
Function: visualize_weights
Description:
    Take an numpy array of shape (n, height, width) or (n, height, width, 3)
    Visualize each (height, width) patch in a grid of size approx. sqrt(n) by sqrt(n)
'''
def visualize_weights(data):
    data = (data - data.min()) / (data.max() - data.min())

    n = int(numpy.ceil(numpy.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))
               + ((0, 0),) * (data.ndim - 3))
    data = numpy.pad(data, padding, mode='constant', constant_values=1)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    from arena.helpers.pretrained import vgg_m
    param = vgg_m()
    conv1 = param['arg:conv1_weight'].asnumpy()
    visualize_weights(conv1.transpose(0, 2, 3, 1))

