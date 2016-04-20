import mxnet as mx
from collections import namedtuple

'''
ImagePatch stores the basic information of the patch the basic attentional element of the tracker
'''

ImagePatch = namedtuple("ImagePatch", ["center", "size", "data"])


class GlimpseHandler(object):
    def __init__(self, scale_mult, depth, output_shape):
        super(GlimpseHandler, self).__init__()
        self.scale_mult = scale_mult
        self.depth = depth
        self.output_shape = output_shape

    '''
    pyramid_glimpse: Generate a spatial pyramid of glimpse sectors, padding zero if necessary
    '''
    def pyramid_glimpse(self, img, center, size, timestamp=0, attention_step=0):
        glimpse = []
        curr_scale = 1.0
        roi = mx.symbol.Concat(center, size, num_args=2, dim=1)
        for i in range(self.depth):
            patch_data = mx.symbol.SpatialGlimpse(data=img, roi=roi,
                                             output_shape=self.output_shape,
                                             scale=curr_scale,
                                             name="glimpse%d(%g)_t%d_step%d"
                                                  %(i, curr_scale, timestamp, attention_step))
            glimpse.append(ImagePatch(center=center, size=size*curr_scale, data=patch_data))
            curr_scale *= self.scale_mult
        return glimpse
