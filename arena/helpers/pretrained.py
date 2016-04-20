import numpy
from arena.utils import *
import mxnet.ndarray as nd
import logging


def vgg_m(vgg_m_path="D:\\HKUST\\mxnet\\tools\\caffe_converter\\VGG_M-0001.params"):
    print 'Loading VGG-M model from %s' %vgg_m_path
    param_dict = nd.load(vgg_m_path)
    print param_dict
    return param_dict


def imagenet_mean(rows, cols):
    return numpy.array([123.68, 116.779, 103.939])  # R, G, B
