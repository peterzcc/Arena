import numpy
from arena.utils import *
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
import cv2
import logging

'''
Function: visualize_weights
Description:
    Take an numpy array of shape (n, height, width) or (n, 3, height, width)
    Visualize each (height, width) patch in a grid of size approx. sqrt(n) by sqrt(n)
'''


def visualize_weights(data, delay=0):
    if 4 == data.ndim:
        data = data.transpose(0, 2, 3, 1)
    data = (data - data.min()) / (data.max() - data.min())
    n = int(numpy.ceil(numpy.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))
               + ((0, 0),) * (data.ndim - 3))
    data = numpy.pad(data, padding, mode='constant', constant_values=1)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose(
        (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    win = cv2.namedWindow("Weight", cv2.WINDOW_NORMAL)
    if 3 == data.ndim:
        cv2.imshow("Weight", data[:, :, ::-1])
    else:
        cv2.imshow("Weight", data[:, :])
    cv2.waitKey(delay)


'''
plot the roi bounding box on the image
im, shape (3, height, width)
roi, normalized version from [-1,1]
'''


def draw_track_res(im, roi, delay=0):
    im = im.transpose(1, 2, 0)
    width = im.shape[1]
    height = im.shape[0]
    roi = (roi + 1) / 2 * [width, height, width, height]
    roi = numpy.uint8(roi)
    pt1 = (roi[0] - roi[2] / 2, roi[1] - roi[3] / 2)
    pt2 = (roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)
    im2 = numpy.zeros(im.shape)
    im2[:] = im
    cv2.rectangle(im2, pt1, pt2, (0, 0, 255), 1)
    win = cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.imshow('Tracking', im2[:, :, ::-1] / 255.0)
    cv2.waitKey(delay)


'''
Simple test for filter visualization and roi drawing
'''
if __name__ == '__main__':
    from arena.iterators import TrackingIterator
    from arena.helpers.pretrained import vgg_m

    param = vgg_m()
    conv1 = param['arg:conv1_weight'].asnumpy()
    visualize_weights(conv1)

    track_iter = TrackingIterator('D:\\HKUST\\2-2\\learning-to-track\\datasets\\OTB100-processed\\otb100-video.lst',
                                  output_size=(240, 320), resize=True)

    data_batch, roi_batch = track_iter.sample(batch_size=32, length=10, interval_step=2)
    print data_batch.shape, roi_batch.shape

    for i in xrange(data_batch.shape[0]):
        for j in xrange(data_batch.shape[1]):
            draw_track_res(data_batch[i, j], roi_batch[i, j], delay=50)
