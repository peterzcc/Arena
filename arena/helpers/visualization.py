# coding: utf-8
from __future__ import absolute_import, division, print_function

import numpy
import mxnet.ndarray as nd
import logging
try:
    import cv2
except ImportError:
    raise ImportError('OpenCV plugin is not installed. '
                      'Some visualization features will be disabled.')
from ..utils import *


class CV2Vis(object):
    _win_reg = dict()

    @staticmethod
    def get_window(name, typ=cv2.WINDOW_NORMAL):
        """Switch to a window

        New window names will be registered in the inner registry

        Parameters
        ----------
        name : str
            Name of the window
        typ :
            cv2.WINDOW_NORMAL

        Returns
        -------

        """
        cv2.namedWindow(name, typ)
        if name not in CV2Vis._win_reg:
            cv2.resizeWindow(winname=name, width=240, height=240)
            CV2Vis._win_reg[name] = typ

    @staticmethod
    def destroy_window(name):
        """Destroy window

        Parameters
        ----------
        name : str
            Name of the window

        Returns
        -------

        """
        assert name in CV2Vis._win_reg, "Window %s not found in the registry!" %name
        cv2.destroyWindow(name)

    @staticmethod
    def get_display_data(data):
        """Transform the input ndarray to the opencv format

        Parameters
        ----------
        data : numpy.ndarray

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
        if 3 == data.ndim:
            data = data[:, :, ::-1]  # Opencv uses BGR
        return data

    @staticmethod
    def display(data, win_name, win_typ=cv2.WINDOW_NORMAL, delay=None,
                save_image=False, save_path=None, save_size=None):
        """Visualize the input data using OpenCV

        Take a numpy array of shape (height, width), (n, height, width) or (n, 3, height, width)
        If the data dimension = 2, the data is directly displayed.
        Otherwise visualize each (height, width) patch in a grid of size approx. sqrt(n) by sqrt(n)

        Parameters
        ----------
        data : numpy.ndarray
        win_name : str
        win_typ :
        delay :
            Set this variable to wait for key event after displaying the image.
            delay <=0 means to wait the key forever
        save_image : bool
            Whether to save the visualization result
        save_path : str or None
            If save_path is not given, image will be saved to the current directory
            and the name will be the window name.
        save_size : tuple or None

        Returns
        -------

        """
        assert 2 <= data.ndim <= 4
        CV2Vis.get_window(win_name, win_typ)
        data = CV2Vis.get_display_data(data)
        cv2.imshow(win_name, data)
        if save_image:
            if save_path is None:
                save_path = os.path.join('.', win_name + '.jpg')
            CV2Vis.save(data, path=save_path, size=save_size)
        if delay is not None:
            cv2.waitKey(delay)

    @staticmethod
    def save(data, path, size=None):
        """

        Parameters
        ----------
        data : numpy.ndarray
        path : str
        size : tuple or None
            (width, height)
        Returns
        -------

        """
        data = CV2Vis.get_display_data(data)
        if size is None:
            size = (480, 480)
        cv2.imwrite(path, cv2.resize(data * 256, size,
                                     interpolation=cv2.INTER_LINEAR))



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
