import mxnet as mx
from collections import namedtuple

'''
AttentionElement is the basic attentional element of the tracker
'''
AttentionElement = namedtuple("AttentionElement", ["attention_center", "attention_size",
                                                   "object_size", "attention_data"])

'''
pyramid_glimpse: Generate a spatial pyramid of glimpse sectors, padding zero if necessary
'''
def pyramid_glimpse(data, roi, initial_scale, scale_multiple, depth, output_shape, timestamp=0):
    glimpses = []
    curr_scale = initial_scale
    if type(roi) is tuple:
        # If roi is a tuple then the roi = (center, size)
        center = roi[0]
        size = roi[1]
        roi = mx.symbol.Concat(*roi, num_args=2, dim=1)
    else:
        center, size = mx.symbol.SliceChannel(roi, num_outputs=2, axis=1)
    for i in range(depth):
        attention_data = mx.symbol.SpatialGlimpse(data=data, roi=roi,
                                          output_shape=output_shape,
                                          scale=curr_scale, name="spatial-glimpse-scale%g-t%d"
                                                                 %(curr_scale, timestamp))
        attention_center = center
        attention_size = size * curr_scale
        object_size = size
        glimpses.append(AttentionElement(attention_data=attention_data,
                                         attention_center=attention_center,
                                         attention_size=attention_size,
                                         object_size=object_size))
        curr_scale *= scale_multiple
    return glimpses

