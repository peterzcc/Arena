from scipy.io import loadmat
import mxnet as mx
import numpy
from arena import Base
from arena.helpers.visualization import *

regularizer = 0.01


embedding_data = loadmat('embeddings.mat')
joint_embedding_data = loadmat('joint-embeddings.mat')
rows, cols, channel_size = embedding_data['xl'].shape
print rows, cols
feature = mx.symbol.Variable("feature")
second_feature = mx.symbol.Variable("second_feature")
gaussian_map = mx.symbol.Variable("gaussian_map")

feature_ffts = mx.symbol.FFT2D(feature)

gaussian_map = mx.symbol.FFT2D(gaussian_map)
gaussian_map = mx.symbol.BroadcastChannel(gaussian_map, dim=1, size=channel_size)

numerator = mx.symbol.ComplexHadamard(gaussian_map, mx.symbol.Conjugate(feature_ffts))
denominator = mx.symbol.ComplexHadamard(mx.symbol.Conjugate(feature_ffts), feature_ffts) + \
              mx.symbol.ComplexHadamard(feature_ffts, mx.symbol.ComplexExchange(feature_ffts))
denominator = mx.symbol.SumChannel(denominator)
denominator = mx.symbol.BroadcastChannel(data=denominator + regularizer, dim=1, size=channel_size)
scores = mx.symbol.ComplexHadamard(numerator / denominator, mx.symbol.FFT2D(second_feature))
scores = mx.symbol.IFFT2D(data=scores, output_shape=(numpy.int32(rows), numpy.int32(cols)))



data_shapes = {'feature': (1, channel_size, rows, cols),
               'second_feature': (1, channel_size, rows, cols),
               'gaussian_map': (1, 1, rows, cols)}
net = Base(data_shapes=data_shapes, sym=scores)
outputs = net.forward(data_shapes=data_shapes, feature=numpy.rollaxis(embedding_data['xl'], 2).reshape((1, channel_size, rows, cols)),
            second_feature=numpy.rollaxis(joint_embedding_data['xt'], 2).reshape((1, channel_size, rows, cols)),
            gaussian_map=numpy.real(numpy.fft.ifft2(embedding_data['yf'])).reshape((1, 1, rows, cols)))
for output in outputs:
    print output.shape
    for i in range(channel_size):
        score = output.asnumpy()[0, i, :, :]
        cv2.imshow('image', score /score.max())
        cv2.waitKey()
        score = numpy.rollaxis(joint_embedding_data['score_map'], 2)[i]
        cv2.imshow('image', score / score.max())
        cv2.waitKey()