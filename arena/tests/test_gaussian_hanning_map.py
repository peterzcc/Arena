import mxnet as mx
from arena import Base
from arena.advanced.tracking import gaussian_map_fft, HannWindowGeneratorOp
import numpy
import cv2

attention_size = mx.symbol.Variable('attention_size')
object_size = mx.symbol.Variable('object_size')

hann_map_op = HannWindowGeneratorOp(rows=64, cols=64)

map_fft = gaussian_map_fft(attention_size=attention_size, object_size=object_size,
                           sigma_factor=10, rows=64, cols=64)
map_recons = mx.symbol.IFFT2D(data=map_fft, output_shape=(64, 64))
map_fft = mx.symbol.BlockGrad(map_fft)
data_shapes = {'attention_size': (1, 2), 'object_size': (1, 2)}
net = Base(sym=map_recons, data_shapes=data_shapes)
output = net.forward(data_shapes=data_shapes, attention_size=numpy.array([[0.5, 0.5]]),
            object_size=numpy.array([[0.3, 0.3]]))[0].asnumpy()
print output.shape
cv2.imshow('image', output[0,0,:,:])
cv2.waitKey()

hann_map = hann_map_op()
net_hann = Base(sym=hann_map, data_shapes=dict())
output = net_hann.forward(data_shapes=dict())[0].asnumpy()
print output.shape
cv2.imshow('image', output[0,0,:,:])
cv2.waitKey()

