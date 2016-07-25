# coding: utf-8
import numpy
import mxnet.ndarray as nd
import cv2
import logging

from ..utils import *

def cv2_visualize(data, delay=None, win_name="Weight", win_typ=cv2.WINDOW_NORMAL,
                  save_path=None):
    """Visualize the input tensor using OpenCV

    Take a numpy array of shape (height, width), (n, height, width) or (n, 3, height, width)
    If the data dimension = 2, the data is directly displayed.
    Otherwise visualize each (height, width) patch in a grid of size approx. sqrt(n) by sqrt(n)

    Parameters
    ----------
    data
    delay
    win_name
    win_typ
    save_path

    Returns
    -------

    """
    assert 2 <= data.ndim <= 4
    if 2 != data.ndim:
        if 4 == data.ndim:
            data = data.transpose(0, 2, 3, 1)
        data = (data - data.min()) / (data.max() - data.min())
        n = int(numpy.ceil(numpy.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                    (0, 0), (0, 0))
                   + ((0, 0),) * (data.ndim - 3))
        data = numpy.pad(data, padding, mode='constant', constant_values=1)

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose(
            (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    win = cv2.namedWindow(win_name, win_typ)
    if 3 == data.ndim:
        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, win_name + '.png'),
                        cv2.resize(data[:, :, ::-1]*256, (480, 480),
                                   interpolation=cv2.INTER_LINEAR))
        cv2.imshow(win_name, data[:, :, ::-1])
    else:
        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, win_name + '.png'),
                        cv2.resize(data[:, :]*256, (480, 480),
                                   interpolation=cv2.INTER_LINEAR))
        cv2.imshow(win_name, data[:, :])
    if delay is not None:
        cv2.waitKey(delay)


def draw_track_res(im, roi, delay=None, color=(0, 0, 255), win_name="Tracking",
                   win_typ=cv2.WINDOW_AUTOSIZE, save_path=None):
    """Plot the ROI bounding box on the image using OpenCV

    Parameters
    ----------
    im : numpy.ndarray
      shape (3, height, width)
    roi : numpy.ndarray
    delay
    color
    win_name
    win_typ
    save_path

    Returns
    -------

    """
    im = im.transpose(1, 2, 0)
    width = im.shape[1]
    height = im.shape[0]
    roi = roi * [width, height, width, height]
    roi = numpy.uint32(roi)
    pt1 = (roi[0] - roi[2] / 2, roi[1] - roi[3] / 2)
    pt2 = (roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)
    im2 = numpy.zeros(im.shape)
    im2[:] = im
    cv2.rectangle(im2, pt1, pt2, color, 1)
    win = cv2.namedWindow(win_name, win_typ)
    cv2.imshow(win_name, im2[:, :, ::-1] / 255.0)
    if save_path is not None:
        cv2.imwrite(os.path.join(save_path, win_name + '.png'), im2[:, :, ::-1])
    if delay is not None:
        cv2.waitKey(delay)


class PLTVisualizer(object):
    def __init__(self, name="PLTVis"):
        self.enabled = True
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.lines, = self.ax.plot([], [])
            self.ax.set_autoscaley_on(True)
            self.fig.canvas.set_window_title(name)
        except(ImportError):
            print('ImportError: matplotlib cannot be imported! PLTVisualizer will be disabled')
            self.enabled = False
            return

    def update(self, new_x, new_y):
        if self.enabled:
            self.lines.set_xdata(numpy.append(self.lines.get_xdata(), new_x))
            self.lines.set_ydata(numpy.append(self.lines.get_ydata(), new_y))
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


'''
Simple test for filter visualization and roi drawing
'''
if __name__ == '__main__':
    from arena.iterators import TrackingIterator
    from arena.helpers.pretrained import vgg_m

    param = vgg_m()
    conv1 = param['arg:conv1_weight'].asnumpy()
    cv2_visualize(conv1)

    track_iter = TrackingIterator('D:\\HKUST\\2-2\\learning-to-track\\datasets\\OTB100-processed\\otb100-video.lst',
                                  output_size=(240, 320), resize=True)

    data_batch, roi_batch = track_iter.sample(batch_size=32, length=10, interval_step=2)
    print data_batch.shape, roi_batch.shape

    for i in xrange(data_batch.shape[0]):
        for j in xrange(data_batch.shape[1]):
            draw_track_res(data_batch.asnumpy()[i, j], roi_batch.asnumpy()[i, j], delay=50)
