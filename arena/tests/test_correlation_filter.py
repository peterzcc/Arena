import mxnet as mx
import mxnet.ndarray as nd
import numpy
import cv2
from arena import Base
import time
import pyfftw
from arena.advanced.attention import GlimpseHandler
from arena.advanced.tracking import CorrelationFilterHandler
from arena.advanced.tracking import PerceptionHandler
from arena.iterators import TrackingIterator
from arena.helpers.visualization import *

sample_length = 20

tracking_iterator = TrackingIterator('D:\\HKUST\\2-2\\learning-to-track\\datasets\\OTB100-processed\\otb100-video.lst', resize=False)
glimpse_handler = GlimpseHandler(scale_mult=1.8, depth=3, output_shape=(133, 133))
perception_handler = PerceptionHandler(net_type='VGG-M')
cf_handler = CorrelationFilterHandler(rows=64, cols=64, gaussian_sigma_factor=10, regularizer=0.01,
                                      perception_handler=perception_handler)

data_images = mx.symbol.Variable('data_images')
data_rois = mx.symbol.Variable('data_rois')
data_rois = mx.symbol.SliceChannel(data_rois, num_outputs=2, axis=2)
data_images = mx.symbol.SliceChannel(data_images, num_outputs=sample_length, axis=1, squeeze_axis=True)
data_centers = mx.symbol.SliceChannel(data_rois[0], num_outputs=sample_length, axis=1, squeeze_axis=True)
data_sizes = mx.symbol.SliceChannel(data_rois[1], num_outputs=sample_length, axis=1, squeeze_axis=True)


multiscale_template_l = []
for i in range(sample_length):
    glimpse_pyramid = glimpse_handler.pyramid_glimpse(img=data_images[i], center=data_centers[i],
                                           size=data_sizes[i], timestamp=i)
    multiscale_template = cf_handler.get_multiscale_template(glimpse=glimpse_pyramid,
                                                             object_size=glimpse_pyramid[0].size,
                                                             timestamp=i)
    multiscale_template_l.append([template.numerator for template in multiscale_template])

templates = mx.symbol.Group(symbols=sum(multiscale_template_l, []))

data_shapes = {'data_images':(1, sample_length, 3, 360, 480), 'data_rois': (1, sample_length, 4)}

net = Base(sym=templates, data_shapes=data_shapes)
perception_handler.set_params(net.params)

start = time.time()
for i in range(10):
    seq_images, seq_rois = tracking_iterator.sample(length=sample_length)
    outputs = net.forward(data_shapes={'data_images': seq_images.shape, 'data_rois': seq_rois.shape},
            data_images=seq_images, data_rois=seq_rois)
    for output in outputs:
        print output.asnumpy().shape
        visualize_weights(output.asnumpy()[0])
end = time.time()
print end-start